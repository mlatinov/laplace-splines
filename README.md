# laplace-splines

A [Laplace](https://github.com/mlatinov/laplace) library of spline building blocks for Stan — B-spline, M-spline, I-spline, natural cubic and tensor-product bases, difference and derivative penalties, non-centred P-spline priors, and helpers for hierarchical (per-group) splines. Import it into any `.laplace` model and call it with namespaced calls (`splines::function_name(...)`).

Like all Laplace libraries, `splines` compiles down to plain, readable Stan functions. Nothing about how you use it hides what actually ends up in your `.stan` file.

## How it's organised

Every equation in the spline literature answers one question: **does this symbol depend on a parameter?** The answer decides which Stan block it lives in, and the library is organised around that split.

| Symbol | Depends on | Stan block |
| --- | --- | --- |
| $x_i$, knots $t_j$, $\xi_j$ | nothing | `data` |
| $B_j(x)$, $\mathbf{B}$, $D_d$, $\Omega$ | data only | `transformed data` |
| $\beta_j$, $\sigma$, $z_j$ | sampled | `parameters` |
| $f = \mathbf{B}\beta$ | both | `transformed parameters` |
| the penalty on $\beta$ | parameters | `model` |

So the library has two halves, and they never mix:

1. **Bases** produce $\mathbf{B}$. They are pure data computations, run once. Boundary conditions, monotonicity and positivity constraints — anything expressed *about the function* — get absorbed here and leave no trace anywhere else in the model.
2. **Penalties** constrain $\beta$. They never touch $\mathbf{B}$. They appear either as a generative recursion in `transformed parameters` or as a quadratic form in `model`.

This is why two things both called "splines" look nothing alike in code. A natural cubic spline is *entirely* a basis construction: once $\mathbf{B}$ is built, the model is one matrix multiply. A P-spline is almost entirely a penalty: its basis is the ordinary B-spline basis, and all the content is in the prior.

Everything reduces to

$$f(x) = \sum_{j=1}^{K}\beta_j B_j(x) \qquad\Longleftrightarrow\qquad \mathbf{f} = \mathbf{B}\beta$$

with the spline types differing only in what the $B_j$ are and what prior sits on $\beta$.

## Knot vectors

Every basis except `ns_basis` needs a knot vector, and there are two conventions. **They are not interchangeable**, and choosing the wrong one silently breaks the P-spline penalty rather than raising an error.

| Function | Convention | Length | $K$ |
| --- | --- | --- | --- |
| `bspline_knots_clamped(a, b, xi, degree)` | A — clamped | $m + 2p + 2$ | $m + p + 1$ |
| `bspline_knots_uniform(a, b, n_seg, degree)` | B — uniform extended | $n_{\text{seg}} + 2p + 1$ | $n_{\text{seg}} + p$ |

**Convention A** repeats each boundary knot $p+1$ times, forcing the basis to interpolate at the edges. This is what R's `splines::bs()` produces. Use it for regression splines, M-splines and I-splines.

**Convention B** lays knots at equal spacing and extends $p$ of them past each end, so every basis function is an exact translate of every other. **P-splines require this.** The difference penalty treats neighbouring coefficients as comparable, which is only justified for identical translates; repeated boundary knots make the penalty mean something different at the edges than in the middle.

In both cases $K = n - p - 1$. Decide which of $K$, $n_{\text{seg}}$ or $m$ your model supplies and derive the rest — don't supply two.

## Bases

| Function | Returns | What it gives you |
| --- | --- | --- |
| `bspline_basis(x, t, degree)` | $N \times K$ | $K$ local bumps that sum to 1 everywhere |
| `mspline_basis(x, t, order)` | $N \times K$ | Same bumps rescaled to unit integral |
| `ispline_basis(x, t, order)` | $N \times K$ | Monotone columns rising $0 \to 1$ |
| `ns_basis(x, xi)` | $N \times K$ | Cubic inside, linear outside the boundary knots |
| `tensor_basis(Bx, Bz)` | $N \times K_xK_z$ | A surface over two covariates |
| `bspline_deriv_basis(x, t, degree, q)` | $N \times K$ | The $q$-th derivative of the basis |
| `center_basis(B)` | $N \times K$ | Any basis, made orthogonal to an intercept |

`mspline_basis` and `ispline_basis` take **order** $=$ degree $+\,1$ (order 4 is cubic), because the literature does. Everything else takes **degree**.

### B-spline

The foundation of everything else. Built bottom-up by the Cox–de Boor recursion:

$$B_{j,q}(x) = \frac{x - t_j}{t_{j+q} - t_j}B_{j,q-1}(x) + \frac{t_{j+q+1} - x}{t_{j+q+1} - t_{j+1}}B_{j+1,q-1}(x)$$

Each basis function is nonzero on only $p+2$ consecutive knot intervals, so coefficients have a direct spatial meaning — which is exactly what makes neighbour-based penalties sensible. The basis satisfies $\sum_j B_j(x) = 1$ for all $x$ in the domain; checking that row sums equal 1 catches nearly every knot-padding error you can make.

### M-spline and I-spline

An **M-spline** is a B-spline rescaled to integrate to 1 rather than sum to 1:

$$M_{j,k}(x) = \frac{k}{t_{j+k} - t_j}B_{j,k-1}(x)$$

Since columns are non-negative, a `simplex[K]` on $\beta$ makes $M\beta$ a valid probability density — the building block for density estimation and baseline hazards.

An **I-spline** is the integral of an M-spline. Each column rises monotonically from 0 to 1, so

$$f(x) = \beta_0 + \sum_j \beta_j I_j(x), \qquad \beta_j \ge 0$$

is non-decreasing **by construction**. The shape constraint lives entirely in the sign of $\beta$ — no penalty, no rejection sampling, no constrained optimisation. For monotone decreasing, use $1 - I$ or flip the sign of $\beta$.

The implementation uses the identity $\int B_{j,p} = \frac{t_{j+p+1}-t_j}{p+1}\sum_{m \ge j}B_{m,p+1}$, whose prefactor cancels the M-spline scaling exactly, leaving $I_j(x) = \sum_{m \ge j} B_{m,k}(x)$ — a plain suffix cumulative sum. No interval lookup, no case analysis.

### Natural cubic

A cubic spline forced to be linear outside the boundary knots, i.e. $f''(x) = 0$ there. Ordinary cubic splines have their highest variance at the edges, where there is least data to constrain them, and extrapolate badly. The natural constraint trades a little flexibility for a large reduction in edge variance.

$$N_1 = 1, \quad N_2 = x, \quad N_{j+2} = d_j - d_{K-1}, \qquad d_j(x) = \frac{(x-\xi_j)^3_+ - (x-\xi_K)^3_+}{\xi_K - \xi_j}$$

Beyond $\xi_K$ every $d_j$ expands to $3x^2 - 3(\xi_K+\xi_j)x + \dots$; the $3x^2$ term is identical for all $j$, so the subtraction cancels it and leaves a straight line. **That subtraction is the boundary condition**, discharged at basis-construction time. Which is why the model code that follows is just `B * beta` and nothing else.

`ns_basis` takes **interior knots directly**, not a knot vector.

### Tensor product

$$\mathbf{B}[i,\ (j-1)K_z + l] = \mathbf{B}^x[i,j]\cdot\mathbf{B}^z[i,l]$$

This is the *row-wise* Kronecker product (Khatri–Rao): the row index is shared between the two factors, so you get one row per observation rather than $N^2$. The stacking order is **$x$ slow, $z$ fast**, and it must match `tensor_penalty_x` / `tensor_penalty_z`.

## Penalties

| Function | Returns | Use it when |
| --- | --- | --- |
| `difference_matrix(K, d)` | $(K-d) \times K$ | You need $D_d$ itself, e.g. to check a fit |
| `penalty_matrix(K, d)` | $K \times K$ | The centred form, $D_d^\top D_d$ |
| `gram_matrix(t, degree, q)` | $K \times K$ | Unequal knots, or penalising `ns_basis` |
| `tensor_penalty_x(Kx, Kz, dx)` | $K_xK_z$ square | Smoothing a surface along $x$ |
| `tensor_penalty_z(Kx, Kz, dz)` | $K_xK_z$ square | Smoothing a surface along $z$ |
| `pspline(head, z, sigma, order)` | $K$ | **The default.** Generative, non-centred |
| `pspline_adaptive(head, z, sigma, order)` | $K$ | Smoothness varies along the covariate |

### The difference penalty, two ways

The P-spline prior has an equivalent penalty form and generative form:

$$p(\beta \mid \sigma) \propto \exp\left[-\frac{1}{2\sigma^2}\beta^\top D_d^\top D_d\,\beta\right] \qquad\Longleftrightarrow\qquad D_d\beta \sim N(0, \sigma^2 I)$$

Non-centring the second and solving for the newest element gives a forward recursion, which is what `pspline` returns:

$$\beta_{j+2} = 2\beta_{j+1} - \beta_j + \sigma z_j \qquad (d = 2)$$

**Prefer `pspline` over the quadratic form.** The quadratic form is the centred parameterisation; it is closer to the published formula and samples considerably worse, typically producing divergences with $\sigma$ stuck near zero. `pspline` is implemented as $d$ nested cumulative sums rather than a scalar loop, so it is also a much shorter autodiff tape.

### Why `pspline` needs a `head`

$D_d$ has rank $K-d$. Its null space — the directions the penalty says *nothing* about — has dimension exactly $d$ and consists of sequences polynomial in the index of degree $< d$:

| $d$ | Null space | `head` | `z` |
| --- | --- | --- | --- |
| 1 | constant | 1 element | $K-1$ |
| 2 | constant + linear | 2 elements | $K-2$ |
| 3 | + quadratic | 3 elements | $K-3$ |

So `head` is not a set of arbitrary starting values. It spans what the penalty leaves unconstrained, and **it needs its own priors** — the penalty will not supply them.

### Derivative penalties

`gram_matrix` penalises the integrated roughness of the *fitted function* rather than differences of coefficients:

$$J(f) = \int_a^b\big[f^{(q)}(x)\big]^2 dx = \beta^\top\Omega\beta, \qquad \Omega_{jl} = \int B_j^{(q)}B_l^{(q)}$$

It is **exact**, not approximate: the integrand is piecewise polynomial of degree $2(p-q)$, and Gauss–Legendre with $G = p-q+1$ nodes integrates degree $\le 2G-1$ exactly.

The trade in one line: difference penalties are cheaper and simpler; derivative penalties are correct for unequal knots, and are the only option for `ns_basis`.

### Adaptive smoothing

Standard P-splines assume the function is equally wiggly everywhere, which is often false — a dose–response curve may be flat at low doses and turn sharply at high ones. `pspline_adaptive` lets the scale vary:

$$(D_d\beta)_j \sim N(0, \sigma_j^2)$$

The $\sigma_j$ must themselves be smooth, or you have added $K-d$ free parameters and learned nothing. The safe construction is a coarse basis on the log scale, $\log\sigma = \mathbf{B}_\lambda\gamma$ with $K_\lambda \approx 5$–$10$.

## Ensemble helpers

For hierarchical splines, `ncp_offset` collapses the per-group assembly loop into a single broadcast. Four overloads, selected by the type of `omega`:

| `omega` | Effect | $W$ |
| --- | --- | --- |
| `real` | One scale for everything | $w_{\text{pop}}\mathbf{1}_J^\top + \omega Z$ |
| `vector[K]` | Scales **rows** — per basis function | $w_{\text{pop}}\mathbf{1}_J^\top + \text{diag}(\omega)\,Z$ |
| `row_vector[J]` | Scales **columns** — per group | $w_{\text{pop}}\mathbf{1}_J^\top + Z\,\text{diag}(\omega)$ |
| `matrix[K, J]` | Fully general | $w_{\text{pop}}\mathbf{1}_J^\top + \Omega \odot Z$ |

These know nothing about splines. `ncp_offset` is "non-centred offset of a matrix of group deviations from a shared vector", and serves GP coefficients or plain random effects equally well.

Every function carries `@brief`, `@param`, `@return`, `@math`, and (where useful) `@example` documentation, so you can read it from the terminal without leaving your model:

```
laplace doc splines::pspline
```

## Installation

`splines` is distributed as a git-hosted Laplace library — there's no published registry entry yet, so it's added by pointing `laplace` (or `cmdlaplacer`, if you're working from R) directly at the repository. The package lives in the repository's `laplace/` subdirectory, so pass it as the subdir.

### Via the `laplace` CLI

From inside a Laplace project (a directory with its own `laplace.toml`):

```
laplace add splines --git https://github.com/mlatinov/laplace-splines --tag 0.1.0 --subdir laplace
```

### Via R (`cmdlaplacer`)

```r
library(cmdlaplacer)

laplace_install_git(
  "splines",
  "https://github.com/mlatinov/laplace-splines",
  tag = "0.1.0",
  subdir = "laplace"
)
```

Either way, this pins the dependency in your project's `laplace.toml`/`laplace.lock` at tag `0.1.0`. Check the [tags](https://github.com/mlatinov/laplace-splines/tags) for newer versions as they become available.

## Usage

Import the library in a `library { }` block and call its functions with the `splines::` namespace prefix.

### A P-spline smooth

The basic case: one smooth term, no grouping. The basis is built once in `transformed data`, and the coefficients are assembled in one line.

```stan
library {
    import splines
}

data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
  int<lower=1> n_seg;                  // number of knot segments, e.g. 20
}

transformed data {
  int degree = 3;
  int K = n_seg + degree;
  vector[n_seg + 2 * degree + 1] t = splines::bspline_knots_uniform(min(x), max(x), n_seg, degree);
  matrix[N, K] B = splines::bspline_basis(x, t, degree);
}

parameters {
  vector[2] head;                      // null space of D_2: level and slope
  vector[K - 2] z;                     // innovations
  real<lower=0> sd_smooth;             // how wiggly the curve may be
  real<lower=0> sigma;
}

transformed parameters {
  vector[K] beta = splines::pspline(head, z, sd_smooth, 2);
}

model {
  z ~ std_normal();                    // structural: part of the parameterisation
  head ~ normal(0, 5);                 // yours: the penalty says nothing here
  sd_smooth ~ normal(0, 1);            // yours: controls smoothness
  sigma ~ std_normal();

  y ~ normal(B * beta, sigma);
}

generated quantities {
  vector[N] f = B * beta;
}
```

There is no separate intercept: the spline absorbs the level, and adding one would leave the model unidentified.

### A hierarchical P-spline

A population curve plus one deviation per group. `pspline` builds the shared coefficients, `ncp_offset` broadcasts the deviations.

```stan
library {
    import splines
}

data {
  int<lower=1> N;
  int<lower=1> J;                      // number of groups
  vector[N] x;
  array[N] int<lower=1, upper=J> group;
  vector[N] y;
  int<lower=1> n_seg;
}

transformed data {
  int degree = 3;
  int K = n_seg + degree;
  vector[n_seg + 2 * degree + 1] t = splines::bspline_knots_uniform(min(x), max(x), n_seg, degree);
  matrix[N, K] B = splines::bspline_basis(x, t, degree);
}

parameters {
  vector[2] head;
  vector[K - 2] z;
  real<lower=0> sd_smooth;
  matrix[K, J] z_group;                // group deviations, non-centred
  real<lower=0> omega;                 // how far groups may depart
  real<lower=0> sigma;
}

transformed parameters {
  vector[K] beta_pop  = splines::pspline(head, z, sd_smooth, 2);
  matrix[K, J] beta_j = splines::ncp_offset(beta_pop, z_group, omega);
}

model {
  z ~ std_normal();
  to_vector(z_group) ~ std_normal();
  head ~ normal(0, 5);
  sd_smooth ~ normal(0, 1);
  omega ~ normal(0, 0.5);
  sigma ~ std_normal();

  y ~ normal(rows_dot_product(B, beta_j[, group]'), sigma);
}
```

With `omega` a `real` and `z_group` iid, the deviations are **white noise in coefficient space** — each group's curve is a smooth population curve plus a rough perturbation. If you want the deviations themselves to be smooth, build them with `pspline` per group instead:

```stan
transformed parameters {
  vector[K] beta_pop = splines::pspline(head, z, sd_smooth, 2);
  matrix[K, J] beta_j;
  for (j in 1:J)
    beta_j[, j] = beta_pop + splines::pspline(head_g[, j], z_group[, j], omega, 2);
}
```

That distinction is the most consequential choice in a hierarchical spline, and it is invisible in the code either way. Decide it deliberately.

### A monotone dose–response curve

I-splines give monotonicity through the sign of $\beta$ alone — there is no penalty term at all.

```stan
library {
    import splines
}

data {
  int<lower=1> N;
  vector[N] dose;
  vector[N] response;
  int<lower=1> n_knot;
}

transformed data {
  int degree = 3;
  int order  = degree + 1;
  vector[n_knot] xi;
  for (i in 1:n_knot)
    xi[i] = min(dose) + i * (max(dose) - min(dose)) / (n_knot + 1);

  vector[n_knot + 2 * degree + 2] t = splines::bspline_knots_clamped(min(dose), max(dose), xi, degree);
  int K = (n_knot + 2 * degree + 2) - order;
  matrix[N, K] I = splines::ispline_basis(dose, t, order);
}

parameters {
  real beta0;                          // baseline response
  vector<lower=0>[K] beta;             // non-negativity IS the constraint
  real<lower=0> sigma;
}

model {
  beta0 ~ normal(0, 5);
  beta  ~ exponential(1);              // mass near zero, so the fit can stay flat
  sigma ~ std_normal();

  response ~ normal(beta0 + I * beta, sigma);
}
```

### From R

With `cmdlaplacer`, the `.laplace` file compiles straight to a `cmdstanr` model, and the generated `.stan` file stays on disk next to it:

```r
library(cmdlaplacer)

mod <- laplace_model("pspline_smooth.laplace")
fit <- mod$sample(
  data = list(N = length(x), x = x, y = y, n_seg = 20)
)

fit$draws("f")
```

## Things to know

- **P-splines need Convention B knots.** Use `bspline_knots_uniform`, never `bspline_knots_clamped`, with `pspline`. The difference penalty assumes basis functions are identical translates; clamped knots break that and the penalty quietly means something different at the edges. Nothing errors.
- **Never put a difference penalty on `ns_basis` coefficients.** `difference_matrix` works because B-spline coefficients are local and ordered along $x$. The natural cubic basis is a global linear part plus truncated cubics, so differencing those coefficients is meaningless. Use `gram_matrix` instead.
- **Build bases in `transformed data`, not `transformed parameters`.** They depend only on data, so they should be computed once. Everything in `transformed parameters` is also written to the output on every draw, which makes enormous CSV files for an $N \times K$ matrix.
- **A spline plus an intercept is not identified.** Because the basis sums to one, adding a constant to every coefficient and subtracting it from the intercept leaves the fit unchanged; NUTS reports poor $\hat{R}$ on both. Drop the intercept, use `sum_to_zero_vector`, or apply `center_basis`. With more than one smooth term, centring every one is not optional.
- **`z ~ std_normal()` is structural, not a prior choice.** It is part of the non-centred parameterisation. `head`, `sigma` and `omega` are the real prior decisions, and the library deliberately leaves them to you.
- **`vector` and `row_vector` `omega` do different things, and when $K = J$ both are legal.** `vector[K]` scales rows (per basis function), `row_vector[J]` scales columns (per group). With equal dimensions the wrong one compiles, runs, and is wrong. Check the declared type at the call site.
- **Test bases with partition of unity.** Row sums of `bspline_basis` output must equal 1 for every interior $x$. This one assertion catches nearly every knot-padding and indexing error, and it is a single line in `generated quantities`.
- **`ns_basis` is ill-conditioned for large $K$.** The truncated power basis has nearly collinear columns as knots multiply. Rescale $x$ to roughly $[0,1]$ and stay below about $K = 15$. R's `ns()` avoids this with a QR construction that is stable but much harder to write.
- **I-spline coefficients need a prior with mass at zero.** `vector<lower=0>[K]` makes the fit monotone, but without shrinkage toward zero it cannot stay flat where the data are flat. Half-normal or exponential.
- **Order versus degree.** `mspline_basis` and `ispline_basis` take *order* $=$ degree $+\,1$, because the literature does. Everything else takes *degree*. Order 4 is cubic.
- **Tensor stacking order is load-bearing.** `tensor_basis` is $x$ slow, $z$ fast, and `tensor_penalty_x` / `tensor_penalty_z` assume the same. Change it in one place and you smooth the wrong direction with no error raised. Note also that $K_xK_z$ grows fast — $20 \times 20$ is 400 parameters.
- **No thin-plate splines.** They need an $N \times N$ eigendecomposition, $O(N^3)$, which is impractical inside Stan above a few thousand observations. Precompute the basis and penalty in R (`mgcv::smoothCon`) and pass them as data.

## License

See [LICENSE](https://github.com/mlatinov/laplace-splines/blob/main/LICENSE).
# Asymptotic Explosion Energy
Code to estimate asymtptotic core-collapse supernova explosion energy following the methods of  [Murphy et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.489..641M/abstract).
Fit a physically motivated functional form for the time dependence of the explosion energy using [emcee](https://emcee.readthedocs.io/en/stable/index.html).
Code formatted with [black-indent](https://github.com/AstroBarker/black-indent), aka my wrapper around [black](https://github.com/psf/black) that isn't a pain about two space indenting.

## Functional Form
We fit a function of the following form

$$E_{\mathrm{expl}} = E_{\infty} - A / t$$

where $$E_{\infty}$$ is the asymptotic explosion energy and $$A$$ is a fit parameter encapsulating physics such as opacitiy and heating rate.
We fit this equation to simulation output using Markov Chain Monte Carlo (MCMC) Bayesian inference methods.
The posterior distribution for $$E_{\infty}$$ and $$A$$ is

$$ P(E_{\infty}, A, \sigma | {E_{\mathrm{sim},i}}) \propto \mathcal{L}({E_{\mathrm{sim},i}} | E_{\infty}, A, \sigma)P(E_{\infty})P(A)P(\sigma) $$

where $$ P(E_{\infty})$$, $$P(A)$$, and $$P(\sigma)$$ are uniform priors.
Variance $$\sigma^2$$ on simulation explosion energies is unknown and treated as a free parameter to be marginalized over.
In practice, it is very small (unless the fit is very bad).

Our likelihoods are given by

$$\mathcal{L}({E_{\mathrm{sim},i}} | E_{\infty}, A, \sigma)P(E_{\infty})P(A)P(\sigma) = \prod_{i} \frac{1}{\sqrt{2\pi\sigma}} e^{-[E_{\mathrm{sim},i} - E_{\mathrm{expl}}(E_{\infty}, a, \sigma)]/[2\sigma^2]}$$

## Usage
With the `src/` contents in your working directory, or in your `$PYTHONPATH`, you may simply
```python
import profiles
mymodel = profiles.Model()
```
which constructs the following model:
```
Collapsar Model Parameters: 

rho0    = 45566325030087.27
alpha   = 2.5
delta   = 3.0
r_cut   = 70
w0      = 0.05
B0      = 3162277660168.3794
entropy = 2.0
M_star  = 2.7837738189772714e+34
R_star  = 40000000000.0
R_s     = 1181300.0304400998
a       = 0.8
M_BH    = 7.953639482792204e+33
r_ISCO  = 3543897.4785131277
```

See `help(profiles.Model)`, or the source code docstrings, for documentation.

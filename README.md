# ASYMPTOTic explosion Energy
Code to estimate asymtptotic core-collapse supernova explosion energy following the methods of  [Murphy et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.489..641M/abstract).
Fit a physically motivated functional form for the time dependence of the explosion energy using [emcee](https://emcee.readthedocs.io/en/stable/index.html).
Code formatted with [black-indent](https://github.com/AstroBarker/black-indent), aka my wrapper around [black](https://github.com/psf/black) that isn't a pain about two space indenting.

## Functional Form
We fit a function of the following form

$E_{\mathrm{expl}} = E_{\infty} - A / t$

where $E_{\infty}$ is the asymptotic explosion energy and $A$ is a fit parameter encapsulating physics such as opacitiy and heating rate.
We fit this equation to simulation output using Markov Chain Monte Carlo (MCMC) Bayesian inference methods.
The posterior distribution for $E_{\infty}$ and $A$ is

$$ P(E_{\infty}, A, \sigma | {E_{\mathrm{sim},i}}) \propto \mathcal{L}({E_{\mathrm{sim},i}} | E_{\infty}, A, \sigma)P(E_{\infty})P(A)P(\sigma) $$

where $P(E_{\infty})$, $P(A)$, and $P(\sigma)$ are uniform priors.
Variance $\sigma^2$ on simulation explosion energies is unknown and treated as a free parameter to be marginalized over.
In practice, it is very small (unless the fit is very bad).

Our likelihoods are given by

$$\mathcal{L}({E_{\mathrm{sim},i}} | E_{\infty}, A, \sigma)P(E_{\infty})P(A)P(\sigma) = \prod_{i} \frac{1}{\sqrt{2\pi\sigma}} e^{-[E_{\mathrm{sim},i} - E_{\mathrm{expl}}(E_{\infty}, a, \sigma)]/[2\sigma^2]}$$

## Usage
With the `src/` contents in your working directory, or in your `$PYTHONPATH`, you may simply
```python
import asymptote
mymodel = profiles.Model(path_to_data, fit_frac)
```

See `help(asymptote.Model)`, or the source code docstrings, for documentation.

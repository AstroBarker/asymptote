# ASYMPTOTic explosion Energy
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Code to estimate asymtptotic core-collapse supernova explosion energy following the methods of  [Murphy et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.489..641M/abstract).
Fit a physically motivated functional form for the time dependence of the explosion energy using [emcee](https://emcee.readthedocs.io/en/stable/index.html).
Code formatted with [black-indent](https://github.com/AstroBarker/black-indent), aka my wrapper around [black](https://github.com/psf/black) that isn't a pain about two space indenting.

## Functional Form
We fit a function of the following form

$$E_{\mathrm{expl}} = E_{\infty} - A / t$$

where $E_{\infty}$ is the asymptotic explosion energy and $A$ is a fit parameter encapsulating physics such as opacitiy and heating rate.
We fit this equation to simulation output using Markov Chain Monte Carlo (MCMC) Bayesian inference methods.
The posterior distribution for $E_{\infty}$ and $A$ is

$$ P(E_{\infty}, A, \sigma | {E_{\mathrm{sim},i}}) \propto \mathcal{L}({E_{\mathrm{sim},i}} | E_{\infty}, A, \sigma)P(E_{\infty})P(A)P(\sigma) $$

where $P(E_{\infty})$, $P(A)$, and $P(\sigma)$ are uniform priors.
Variance $\sigma^2$ on simulation explosion energies is unknown and treated as a free parameter to be marginalized over.
In practice, it is very small (unless the fit is very bad).

Our likelihoods are given by

$$\mathcal{L}({E_{\mathrm{sim},i}} | E_{\infty}, A, \sigma)P(E_{\infty})P(A)P(\sigma) = \prod_{i} \frac{1}{\sqrt{2\pi\sigma}} e^{-[E_{\mathrm{sim},i} - E_{\mathrm{expl}}(E_{\infty}, a, t)]^2/[2\sigma^2]}$$

In practice, you should fit the latter half or third of your data (post shock revival). 
This is controlled with the `--frac` command line arg or `self.frac` member attribute. 
Default value is 0.5, to only fit the second half of data, but this may need tweaking on a case-by-case basis.

## Dependencies
- numpy
- matplotlib
- emcee
- corner

## Usage
This code assumes that it is loading FLASH `.dat` output. 
To use with your data, modify `load_expl_energy_()` as needed to load time and energy data.

With `asymptote.py` contents in your working directory, or in your `$PYTHONPATH`, you may simply
```python
import asymptote
mymodel = asymptote.Model(path_to_data, fit_frac)
mymodel.Fit_Energy(nwalkers=32, nsamples=16384, nburn=512)
E_asym = mymodel.E_asym
```

or simply as a script
```shell
python asymptote.py --nwalkers 32 --nsamples 16384 --nburn 512 --frac 0.5
```

See `help(asymptote.Model)`, or the source code docstrings, for documentation.

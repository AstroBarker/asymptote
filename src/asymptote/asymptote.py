"""
Purpose: Estimate asymptotic explosion energies with MCMC methods.
Author: Brandon Barker
Inspiration: Murphy et al 2019 (ADS 2019MNRAS.489..641M)
"""

#            _______     ____  __ _____ _______ ____ _______ ______
#     /\    / ____\ \   / /  \/  |  __ \__   __/ __ \__   __|  ____|
#    /  \  | (___  \ \_/ /| \  / | |__) | | | | |  | | | |  | |__
#   / /\ \  \___ \  \   / | |\/| |  ___/  | | | |  | | | |  |  __|
#  / ____ \ ____) |  | |  | |  | | |      | | | |__| | | |  | |____
# /_/    \_\_____/   |_|  |_|  |_|_|      |_|  \____/  |_|  |______|

import argparse
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import warnings


def load_expl_energy_flash(fn):
  """
  load explosion energy from FLASH .dat output
  """

  t, energy = np.loadtxt(fn, usecols=(0, 9), unpack=True)
  return t, energy


# End load_expl_energy_flash


class Model:
  """
  Class for holding explosion energy and fitting routines.
  Purpose: fit asymptotic explosion energy. See Murphy et al 2019 (ADS 2019MNRAS.489..641M)

  Args:
    t (np.array): Time array.
    expl_energy (np.array): Explosion energy array.

  Attributes:
    fit_frac (float): fraction of explosion energy to fit (0.0, 1.0).
    Fit the final fit_frac of data. Default: 0.5
    flat_samples (np.array): MCMC output.
    E_asym (float): Asymptotic explosion energy.
    A (float): Fit parameter A.
    E_error (np.array): Uncertainties E = E_asym [+ error_above, - error_below]
    A_error (np.array): Uncertainties on fit parameter A.

  Methods:
    model_e_expl_: fit function for MCMC.
    model_e_expl_error_: analytic error for fit function.
    log_likelihood_: Log likelihood for MCMC.
    log_prior_: Uniform priors.
    log_propability_: Combied prior and likelihood.
    fit_energy: Run MCMC sampler
    plot_corner: produce a MCMC corner plot.
    plot_energy: plot explosion energy with extrapolation.
    propagate_error: Monte Carlo error porpagation for log(E) -> E

  Usage:
    >>> model = Model(time, energy, fit_frac)
    >>> model.fit_energy(
    ...   nwalkers=args.nwalkers, nsamples=args.nsamples, nburn=args.nburn
    ... )
    >>> E_asym = model.E_asym

    or as a script:
    >>> python fit_energy.py path_to_data_file --nwalkers 16 --nsamples 2048 --nburn 128 --frac 0.25
  """

  def __init__(self, time, expl_energy, frac=0.5):
    # --- Do some input checking ---

    # Check. Frac must be in (0.0, 1.0)
    if frac <= 0.0 or frac >= 1.0:
      raise ValueError("frac must be in open interval (0.0, 1.0)")

    # check on input data dims
    if len(time) != len(expl_energy):
      raise ValueError(
        "Explosion energy and time arrays have different lengths"
      )

    # Hack check: ensure energy is in ergs
    large_energy = 100.0e51
    large_log_energy = np.log10(large_energy)  # log(100 foe)
    if np.max(expl_energy) < large_log_energy:
      raise ValueError(
        "Explosion energies are weird: are they in the correct units? (ergs)"
      )

    self.fit_frac = frac
    self.t = time
    self.expl_energy = expl_energy

    self.flat_samples = None  # MCMC data, hold onto for making posterior plots.
    self.E_asym = 0.0  # target quantity
    self.A = 0.0
    self.E_error = np.array([0.0, 0.0])
    self.A_error = np.array([0.0, 0.0])

  # End __init__

  def model_e_expl_(self, E_inf, A, t):
    """
    Simple function form for the explosion energy assuming
    a neutrino driven mechanism. See Murphy et al 2019 (https://arxiv.org/pdf/1904.09444.pdf)

    E_expl(t) = E_inf - A / t

    We seek to fit E_inf.
    """

    return E_inf - A / t

  # End model_e_expl_

  def model_e_expl_error_(self, sigma_E, sigma_A, t):
    """
    propagate error in to explosion energy curve, for plotting.
    Asymptotes to the error on the asymptotic explosion energy.
    """

    return sigma_E * sigma_E + sigma_A * sigma_A / (t * t)

  # End model_e_expl_error_

  def propagate_error(self):
    """
    Apply Monte Carlo error propagation for log(E) -> E
    """
    if self.E_error[0] == 0.0:
      raise ValueError("Energy uncertainty is 0.0. Have you ran the MCMC?")
    if self.flat_samples is None:
      raise ValueError("MCMC posteriors are empty. Have you ran the MCMC?")

    # can probably just energies = flat_samples[:,0]... oh well.
    n_mc_samples = 3 * len(self.flat_samples[:, 0])
    energies = np.random.choice(self.flat_samples[:, 0], n_mc_samples)

    vals = np.power(10.0, energies)  # log(E) -> E
    error = np.percentile(vals, [16, 50, 84])
    mean = error[1]
    error_below = mean - error[0]
    error_above = error[2] - mean

    return mean, error_above, error_below

  # End propagate_error

  def log_likelihood_(self, theta, t, y):
    """
    Log likelihood function for fitting explosion energy at infinity
    """

    E_inf, A, sigma = theta
    y_model = self.model_e_expl_(E_inf, A, t)
    resid = y - y_model

    return -0.5 * np.sum((resid**2) / sigma + np.log(sigma))

  # End log_likelihood_

  def log_prior_(self, theta):
    """
    flat priors
    """
    E_inf, A, sigma = theta
    if sigma <= 0.0 or A <= 0.0 or E_inf <= 0.0 or E_inf >= 60.0:
      return -np.inf

    return 0.0

  # End log_prior_

  def log_probability_(self, theta, x, y):
    """
    log probability
    """
    lp = self.log_prior_(theta)
    if not np.isfinite(lp):
      return -np.inf
    return lp + self.log_likelihood_(theta, x, y)

  # End log_probability_

  def fit_energy(self, nwalkers=2**4, nsamples=2**14, nburn=2**12):
    """
    MCMC fitting of asymptotic explosion energy
    """
    ndim = 3

    # ! only fit part of data !
    ind = np.max(np.where(self.t <= (1.0 - self.fit_frac) * self.t[-1]))
    t = self.t[ind:]
    expl_energy = self.expl_energy[ind:]

    x = t
    y = np.log10(expl_energy)  # work in log space

    pos = np.zeros((nwalkers, ndim))
    # Random guess for initial position around final explosion energy
    mu = y[-1]
    sigma = np.e
    pos[:, 0] = np.random.normal(mu, sigma, nwalkers)

    # sample A, sigma uniform in open intervals (0.0, e) and (0.0, 0.1)
    eps = np.finfo(float).eps
    pos[:, 1] = np.random.uniform(eps, np.e, nwalkers)
    pos[:, 2] = np.random.uniform(eps, 0.1, nwalkers)

    sampler = emcee.EnsembleSampler(
      nwalkers, ndim, self.log_probability_, args=(x, y)
    )
    sampler.run_mcmc(pos, nsamples, progress=True)
    self.flat_samples = sampler.get_chain(discard=nburn, thin=15, flat=True)

    self.E_asym = np.percentile(self.flat_samples[:, 0], 50)
    self.A = np.percentile(self.flat_samples[:, 1], 50)
    self.E_error[0] = np.percentile(self.flat_samples[:, 0], 84) - self.E_asym
    self.E_error[1] = self.E_asym - np.percentile(self.flat_samples[:, 0], 16)
    self.A_error[0] = np.percentile(self.flat_samples[:, 1], 84) - self.A
    self.A_error[1] = self.A - np.percentile(self.flat_samples[:, 1], 16)

  # End fit_energy

  def plot_corner(self):
    """
    Produce corner plot of MCMC posteriors
    """
    labels = [r"$E_{\infty}$", r"$A$", r"$\sigma^2$"]

    corner.corner(
      self.flat_samples[:, :4],
      labels=labels,
      quantiles=(0.16, 0.5, 0.84),
      show_titles=True,
      title_fmt=".6f",
    )
    plt.savefig("cornerplot.png")

  # End plot_corner

  def plot_energy(self):
    """
    Plot explosion energy vs time with extrapolation
    """
    t_end = 3.0  # plot till t_end

    f = 1.0 - self.fit_frac
    time = np.linspace(f * self.t[-1], t_end, 5000)
    energy = self.model_e_expl_(self.E_asym, self.A, time)

    fig, ax = plt.subplots()

    # add machine eps to expl_energy to avoid log10(0)
    eps = np.finfo(float).eps
    ax.plot(self.t, np.log10(self.expl_energy + eps), color="rebeccapurple")

    error = np.sqrt(
      self.model_e_expl_error_(
        np.mean(self.E_error), np.mean(self.A_error), time
      )
    )
    ax.fill_between(
      time, energy - error, energy + error, color="rebeccapurple", alpha=0.25
    )

    ax.set(
      ylim=[50, self.E_asym * 1.005],
      xlabel="Time [s]",
      ylabel=r"log$_{10}$(E$_{\mathrm{expl}}$)",
    )
    plt.savefig("energy.png")

  # End plot_energy


def main():
  # Instantiate the parser
  parser = argparse.ArgumentParser(
    prog="asymptote",
    description="Asymptotic explosion energy estimator with MCMC methods",
  )

  # Required positional argument
  parser.add_argument("filename", type=str, help="Filename of data file")

  # Optional argument
  parser.add_argument(
    "--nwalkers",
    type=int,
    default=2**4,
    help="Number of MCMC walkers. Should at least be 6 for this MCMC algorithm.",
  )
  parser.add_argument(
    "--nsamples", type=int, default=2**14, help="MCMC walker chain length"
  )
  parser.add_argument(
    "--nburn", type=int, default=2**12, help="MCMC burn length"
  )
  parser.add_argument(
    "--frac",
    type=float,
    default=0.5,
    help="Fit last fraction of data. Default: 0.5",
  )

  args = parser.parse_args()
  filename = args.filename
  fraction = args.frac

  # load explosion energy data
  # NOTE: Change this for other simulation data.
  # NOTE: Should be in ergs
  time, expl_energy = load_expl_energy_flash(filename)

  # --- Do some input checking ---

  if args.nwalkers < 6:
    args.nwalkers = 6
    warnings.warn(
      "\nWarning: nwalkers is less than 6. "
      + "This will cause issues in the MCMC stepping algorithm. Setting nwalkers to 6.\n"
    )

  # make sure nburn is less than nwalkers
  if args.nburn >= args.nsamples:
    raise ValueError("nburn is too large compared to nsamples.")

  # print(help(Model))

  # --- Run the MCMC model ---

  model = Model(time, expl_energy, fraction)
  model.fit_energy(
    nwalkers=args.nwalkers, nsamples=args.nsamples, nburn=args.nburn
  )
  model.plot_corner()
  model.plot_energy()

  print("=========================================================")
  print(f"Asymptotic explosion energy: log(E) = {model.E_asym:.5f}")
  print(f"Uncertainties: +{model.E_error[0]:.5e}, -{model.E_error[1]:.5e}")
  print("=========================================================\n")

  # propagate error log(E) -> E
  mean_e, plus, minus = model.propagate_error()

  print("=========================================================")
  print(f"Asymptotic explosion energy: E = {mean_e:.5e} erg")
  print(f"Uncertainties: +{plus:.5e} erg, -{minus:.5e} erg")
  print("=========================================================")


# End main

if __name__ == "__main__":
  main()

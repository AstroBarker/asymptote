#!/usr/bin/env python3

import sys

sys.path.append("src/")

import numpy as np
import matplotlib.pyplot as plt

import asymptote


def model_e_expl(E_inf, A, t):
  """
  Simple function form for the explosion energy assuming
  a neutrino driven mechanism. See Murphy et al 2019 (https://arxiv.org/pdf/1904.09444.pdf)

  E_expl(t) = E_inf - A / t

  We seek to fit E_inf.
  """

  return E_inf - A / t


def test_asymptote():
  """
  Test asymptote iteration
  """

  fptol = 1.0e-4

  # create synthetic data
  E_inf_fake = np.log10(1.11e51)
  A_fake = 0.05
  t_start = 0.05
  t_end = 3.0
  n = 10000
  t_fake = np.linspace(t_start, t_end, n)
  e_expl_fake = model_e_expl(E_inf_fake, A_fake, t_fake)

  # add some little noise
  noise_mu = 0.00001
  noise_sigma = noise_mu / 500.0
  noise = np.random.normal(noise_mu, noise_sigma, n)

  e_expl_fake += noise

  # set up and run MCMC
  nwalkers = 16
  nsamples = 12000
  nburn = 3000
  fraction = 0.8  # high since it is synthetic data

  model = asymptote.Model(t_fake, np.power(10.0, e_expl_fake), fraction)
  model.fit_energy(nwalkers=nwalkers, nsamples=nsamples, nburn=nburn)

  my_ans = model.E_asym

  ANS = E_inf_fake

  assert abs(my_ans - ANS) < fptol, f"Root must be within {fptol}"


# End test_asymptote


if __name__ == "__main__":
  # main
  test_asymptote()

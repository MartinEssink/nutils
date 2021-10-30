#! /usr/bin/env python3
#
# In this script we solve the Cahn-Hilliard equation, which models the unmixing
# of two phases (φ=+1 and φ=-1) under the influence of surface tension. It is a
# mixed equation of two scalar equations for phase φ and chemical potential η:
#
#     dφ/dt = -div(J)
#     η = ψ'(φ) σ / ε - σ ε Δ(φ)
#
# along with constitutive relations for the flux vector J = -M ∇η and the
# double well potential ψ = .5 (φ² - 1)², and subject to boundary conditions
# ∇φ·n = -σd / σ ε and ∇η·n = 0. Parameters are the interface thickness ε,
# fluid surface tension σ, differential wall surface tension σd, and mobility M.
#
# Cahn-Hilliard is a diffuse interface model, which means that phases do not
# separate sharply, but instead form a transition zone between the phases. The
# transition zone has a thickness proportional to ε, as is readily confirmed in
# one dimension, where a steady state solution on an infinite domain is formed
# by η(x) = 0, φ(x) = tanh(x / ε).
#
# The main driver of the unmixing process is the double well potential ψ that
# is proportional to the mixing energy, taking its minima at the pure phases
# φ=+1 and φ=-1. The interface between the phases adds an energy contribution
# proportional to its length. At the wall we have a phase-dependent fluid-solid
# energy. Over time, the system minimizes the total energy:
#
#     E(φ) = ∫_Ω ψ(φ) σ / ε + ∫_Ω .5 σ ε |∇φ|² + ∫_Γ (σm + φ σd)
#            \                \                   \
#             mixing energy    interface energy    wall energy
#
# Proof: the time derivative of E followed by substitution of the strong form
# and boundary conditions yields dE/dt = ∫_Ω η dφ/dt = -∫_Ω M |∇η|² ≤ 0. □
#
# Switching to discrete time we set dφ/dt = (φ - φ0) / dt and add a stabilizing
# perturbation term ψm(φ, φ0) (φ - φ0) to the doube well potential for reasons
# outlined below. This yields the following time discrete system:
#
#     φ = φ0 - dt div(J)
#     η = (ψ'(φ) + ψm(φ, φ0) (φ - φ0)) σ / ε - σ ε Δ(φ)
#
# For stability we wish for the perturbation ψm to be such that the time
# discrete system preserves the energy dissipation property:
#
#     E(φ) - E(φ0) = ∫_Ω (1 - φ² - .5 (φ + φ0)² - ψm(φ, φ0)) (φ - φ0)² σ / ε
#                  - ∫_Ω (.5 σ ε |∇φ - ∇φ0|² + dt M |∇η|²) ≤ 0
#
# The inequality holds if the perturbation term is bounded from below such that
# ψm(φ, φ0) ≥ 1 - φ² - .5 (φ + φ0)². To keep the energy minima at the pure
# phases we additionally impose that ψm(±1, φ0) = 0, and select 1 - φ² as a
# suitable upper bound which we will refer to as "nonlinear".
#
# We next observe that η is linear in φ if ψm(φ, φ0) = α(φ0) - φ² - (φ + φ0)²
# for any function α, which dominates if α(φ0) ≥ 1 + .5 (φ + φ0)². While this
# cannot be made to hold true for all φ, we make it hold for -√2 ≤ φ, φ0 ≤ √2
# by defining α(φ0) = 2 + 2 |φ0| + φ0², which we will refer to as "linear".
# This scheme further satisfies a weak minima preservation ψm(±1, ±|φ0|) = 0.
#
# We have thus arrived at the three stabilization schemes implemented here:
#
# - Nonlinear: ψm(φ, φ0) = 1 - φ²
# - Linear: ψm(φ, φ0) = 2 + 2 |φ0| - 2 φ (φ + φ0)
# - None: ψm(φ, φ0) = 0 (not stabilized, violates dissipation inequality)
#
# The stab enum in this script defines the schemes in the form of the
# anti-derivative of ψm(φ, φ0) (φ - φ0) to allow Nutils to construct the
# residuals through automatic differentiation.

from nutils import mesh, function, solver, numeric, export, cli, testing
from nutils.expression_v2 import Namespace
from nutils.SI import Length, Time, Density, Tension, Energy, Pressure, Velocity, Quantity as Q
import numpy, itertools, enum
import treelog as log

class stab(enum.Enum):
  none = '0'
  linear = 'dφ^2 (1 + abs(φ0) - .5 (φ + φ0)^2)'
  nonlinear = 'dφ^2 (.5 - .5 φ^2 + φ dφ / 3 - dφ^2 / 12)'

def main(size:Length, epsilon:Length, mobility:Time/Density, stens:Tension,
         wtensn:Tension, wtensp:Tension, nelems:int, etype:str, btype:str,
         degree:int, timestep:Time, tol:Energy/Length, endtime:Time, seed:int,
         circle:bool, stab:stab):
  '''
  Cahn-Hilliard equation on a unit square/circle.

  .. arguments::

     size [10cm]
       Domain size.
     epsilon [2mm]
       Interface thickness; defaults to an automatic value based on the
       configured mesh density if left unspecified.
     mobility [1μL*s/kg]
       Mobility.
     stens [50mN/m]
       Surface tension.
     wtensn [30mN/m]
       Wall surface tension for phase -1.
     wtensp [20mN/m]
       Wall surface tension for phase +1.
     nelems [0]
       Number of elements along domain edge. When set to zero a value is set
       automatically based on the configured domain size and epsilon.
     etype [square]
       Type of elements (square/triangle/mixed).
     btype [std]
       Type of basis function (std/spline), with availability depending on the
       configured element type.
     degree [2]
       Polynomial degree.
     timestep [1min]
       Time step.
     tol [1nJ/m]
       Newton tolerance.
     endtime [2h]
       End of the simulation.
     seed [0]
       Random seed for the initial condition.
     circle [no]
       Select circular domain as opposed to a unit square.
     stab [linear]
       Stabilization method (linear/nonlinear/none).
  '''

  nmin = round(size / epsilon + .5)
  if nelems <= 0:
    nelems = nmin
    log.info('setting nelems to {}'.format(nelems))
  elif nelems < nmin:
    log.warning('mesh is too coarse, consider increasing nelems to {:.0f}'.format(nmin))

  log.info('contact angle: {:.0f}°'.format(numpy.arccos((wtensn - wtensp) / stens) * 180 / numpy.pi))

  domain, geom = mesh.unitsquare(nelems, etype)
  if circle:
    angle = (geom-.5) * (numpy.pi/2)
    geom = .5 + function.sin(angle) * function.cos(angle)[[1,0]] / numpy.sqrt(2)

  bezier = domain.sample('bezier', 5) # sample for surface plots
  grid = domain.locate(geom, numeric.simplex_grid([1,1], 1/40), maxdist=1/nelems, skip_missing=True, tol=1e-5) # sample for quivers

  φbasis = ηbasis = domain.basis('std', degree=degree)
  ηbasis *= stens / epsilon # basis scaling to give η the required unit

  ns = Namespace()
  ns.x = size * geom
  ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
  ns.ε = epsilon
  ns.σ = stens
  ns.φ = function.dotarg('φ', φbasis)
  ns.σmean = (wtensp + wtensn) / 2
  ns.σdiff = (wtensp - wtensn) / 2
  ns.σwall = 'σmean + φ σdiff'
  ns.φ0 = function.dotarg('φ0', φbasis)
  ns.dφ = 'φ - φ0'
  ns.η = function.dotarg('η', ηbasis)
  ns.ψ = '.5 (φ^2 - 1)^2'
  ns.dψ = stab.value
  ns.M = mobility
  ns.J_i = '-M ∇_i(η)'
  ns.dt = timestep

  nrg_mix = domain.integral('(ψ σ / ε) dV' @ ns, degree=7)
  nrg_iface = domain.integral('.5 σ ε ∇_k(φ) ∇_k(φ) dV' @ ns, degree=7)
  nrg_wall = domain.boundary.integral('σwall dS' @ ns, degree=7)
  nrg = nrg_mix + nrg_iface + nrg_wall + domain.integral('(dψ σ / ε - η dφ + .5 dt J_k ∇_k(η)) dV' @ ns, degree=7)

  numpy.random.seed(seed)
  state = dict(φ=numpy.random.normal(0, .5, φbasis.shape)) # initial condition

  with log.iter.fraction('timestep', range(round(endtime / timestep))) as steps:
   for istep in steps:

    E = numpy.stack(function.eval([nrg_mix, nrg_iface, nrg_wall], **state))
    log.user('energy: {0:,.0μJ/m} ({1[0]:.0f}% mixture, {1[1]:.0f}% interface, {1[2]:.0f}% wall)'.format(numpy.sum(E), 100*E/numpy.sum(E)))

    state['φ0'] = state['φ']
    state = solver.optimize(['φ', 'η'], nrg / tol, arguments=state, tol=1)

    x, φ = bezier.eval(['x_i', 'φ'] @ ns, **state)
    x_, J = grid.eval(['x_i', 'J_i'] @ ns, **state)
    with export.mplfigure('phase.png') as fig:
      ax = fig.add_subplot(aspect='equal', xlabel='[mm]', ylabel='[mm]')
      im = ax.tripcolor(*x.T/Q('mm'), bezier.tri, φ, cmap='bwr')
      im.set_clim(-1, 1)
      fig.colorbar(im)
      ax.quiver(*x_.T/Q('mm'), *J.T.magnitude, color='r')
      ax.quiver(*x_.T/Q('mm'), *-J.T.magnitude, color='b')
      ax.autoscale(enable=True, axis='both', tight=True)

    log.info('largest flux: {:.1mm/h}'.format(numpy.max(numpy.hypot(*J.T))))

  return state

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line.

if __name__ == '__main__':
  cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

  @testing.requires('matplotlib')
  def test_initial(self):
    state = main(size=Q('10cm'), epsilon=Q('5cm'), mobility=Q('1μL*s/kg'),
      stens=Q('50mN/m'), wtensn=Q('30mN/m'), wtensp=Q('20mN/m'), nelems=3,
      etype='square', btype='std', degree=2, timestep=Q('1h'), tol=Q('1nJ/m'),
      endtime=Q('1h'), seed=0, circle=False, stab=stab.linear)
    with self.subTest('concentration'): self.assertAlmostEqual64(state['φ0'], '''
      eNoBYgCd/xM3LjTtNYs3MDcUyt41uc14zjo0LzKzNm812jFhNNMzwDYgzbMzV8o0yCM1rzWeypE3Tcnx
      L07NzTa4NlMyETREyrPIGMxYMl82VDbjy1/M8clZyf3IRjday6XLmMl6NRnJMF4tqQ==''')

  @testing.requires('matplotlib')
  def test_square(self):
    state = main(size=Q('10cm'), epsilon=Q('5cm'), mobility=Q('1μL*s/kg'),
      stens=Q('50mN/m'), wtensn=Q('30mN/m'), wtensp=Q('20mN/m'), nelems=3,
      etype='square', btype='std', degree=2, timestep=Q('1d'), tol=Q('1nJ/m'),
      endtime=Q('2d'), seed=0, circle=False, stab=stab.linear)
    with self.subTest('concentration'): self.assertAlmostEqual64(state['φ'], '''
      eNoBYgCd/x82/TWKNek08DP2MaUy/DXZNVk1oDRsM8wvZTG1NYo1xzRmM+cnrMwizWA1KjXxMxMvk8wr
      y1/LHTXcNAsz7c2Vy5fKv8rUNIY0gjF1zPjKLspOygo1xDR2MvPMMMtSynXKrIgvAQ==''')
    with self.subTest('chemical-potential'): self.assertAlmostEqual64(state['η'], '''
      eNoBYgCd/+bL98s4zGvMzstDyy/L58v4yzHMW8y7yy/LHMv9ywjM2MuZy/vKfMpwyhbMG8yNyxDLdMr9
      yfXJEcwRzFbLwsoxysLJvMkBzP7LJcuDyvvJk8mPyfvL98siy4PK/MmUyZDJbKtA8A==''')

  @testing.requires('matplotlib')
  def test_mixedcircle(self):
    state = main(size=Q('10cm'), epsilon=Q('5cm'), mobility=Q('1μL*s/kg'),
      stens=Q('50mN/m'), wtensn=Q('30mN/m'), wtensp=Q('20mN/m'), nelems=3,
      etype='mixed', btype='std', degree=2, timestep=Q('1d'), tol=Q('1nJ/m'),
      endtime=Q('2d'), seed=0, circle=True, stab=stab.linear)
    with self.subTest('concentration'): self.assertAlmostEqual64(state['φ'], '''
      eNoBYgCd/6M0sjSPNJE0qTRzNEw0PTTJM2Y08TNKNFEzwDAiM1Az/DHPMass5M7wzsoztjKRMO0zczO5
      MtQvWs7gz7nNqsxgzUXNzczEMu0wbNHdMUwwCCzbzWzN7M76z8nMMc27zXzNq5Uyjg==''')
    with self.subTest('chemical-potential'): self.assertAlmostEqual64(state['η'], '''
      eNoBYgCd/4/M1syyzEnM3MzPzNDLD8wTzKnM7svpzKbLE8sEzHLL2crPymzKUcoxyn3LeMv2ylzLEcvx
      yonLocoSy5vKAMr6yQnKLMrPyp7Kr8q7yq3Kucq1ypTKuMrIyl/KM8qAylnKm0JAzQ==''')

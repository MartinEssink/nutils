#! /usr/bin/env python3
#
# In this script we solve the linear elasticity problem on a unit square
# domain, clamped at the left boundary, and stretched at the right boundary
# while keeping vertical displacements free.

from nutils import mesh, function, solver, export, cli, testing
from nutils.expression_v2 import Namespace
import numpy

def main(nelems:int, etype:str, btype:str, degree:int, poisson:float):
  '''
  Horizontally loaded linear elastic plate.

  .. arguments::

     nelems [10]
       Number of elements along edge.
     etype [square]
       Type of elements (square/triangle/mixed).
     btype [std]
       Type of basis function (std/spline), with availability depending on the
       configured element type.
     degree [1]
       Polynomial degree.
     poisson [.25]
       Poisson's ratio, nonnegative and strictly smaller than 1/2.
  '''

  domain, geom = mesh.unitsquare(nelems, etype)

  ns = Namespace()
  ns.δ = function.eye(domain.ndims)
  ns.x = geom
  ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
  ns.basis = domain.basis(btype, degree=degree)
  ns.u = function.dotarg('u', ns.basis, shape=(2,))
  ns.t = function.dotarg('t', ns.basis, shape=(2,))
  ns.X_i = 'x_i + u_i'
  ns.μ = .5/poisson - 1
  ns.E = '∇_i(u_i) ∇_j(u_j) + μ ∇_i(u_j) ∇_j(u_i) + μ ∇_i(u_j) ∇_i(u_j)'

  sqr = domain.boundary['left'].integral('u_k u_k dS' @ ns, degree=degree*2)
  cons = solver.optimize(['u'], sqr, droptol=1e-15)

  energy = domain.integral('E dV' @ ns, degree=degree*2)
  work = domain.boundary['right'].integral('u_i n_i dS' @ ns, degree=degree*2)
  state = solver.optimize(['u'], energy - work, constrain=cons)

  clamp_work = domain.boundary['left'].integral('u_i t_i dS' @ ns, degree=degree*2)
  invcons = dict(t=numpy.choose(numpy.isnan(cons['u']), [numpy.nan, 0.]))
  state = solver.solve_linear(['t'], [(energy - clamp_work).derivative('u')], constrain=invcons, arguments=state)

  bezier = domain.sample('bezier', 5)
  X, E = bezier.eval(['X_i', 'E'] @ ns, **state)
  Xt, t = domain.boundary['left'].sample('uniform', 3).eval(['X_i', 't_i'] @ ns, **state)
  with export.mplfigure('shear.png') as fig:
    ax = fig.add_subplot(111, xlim=(-.25,1.25), aspect='equal')
    im = ax.tripcolor(*X.T, bezier.tri, E, shading='gouraud', rasterized=True)
    ax.quiver(*Xt.T, *t.T)
    fig.colorbar(im)

  return cons, state

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# elasticity.py`. To select mixed elements and quadratic basis functions add
# :sh:`python3 elasticity.py etype=mixed degree=2`.

if __name__ == '__main__':
  cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

  @testing.requires('matplotlib')
  def test_default(self):
    cons, state = main(nelems=4, etype='square', btype='std', degree=1, poisson=.25)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons['u'], '''
      eNpjYMACGqgLASCRFAE=''')
    with self.subTest('displacement'): self.assertAlmostEqual64(state['u'], '''
      eNpjYMAEBYYKBkqGMXqyhgwMSoZLLhYYPji/wajBYI6Rjv4kIwaGOUZXLgD550uNpxvkG7voZxszMOQb
      77lQapx5ns1kkQGzSZA+owkDA7PJugtsJnHnATXSGpw=''')
    with self.subTest('traction'): self.assertAlmostEqual64(state['t'], '''
      eNoTObHnJPvJeoP9JxgY2E82nhc54WLGQGUAACftCN4=''')

  @testing.requires('matplotlib')
  def test_mixed(self):
    cons, state = main(nelems=4, etype='mixed', btype='std', degree=1, poisson=.25)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons['u'], '''
      eNpjYICCBiiEsdFpCiAARJEUAQ==''')
    with self.subTest('solution'): self.assertAlmostEqual64(state['u'], '''
      eNpjYICAPEMhAy1DBT0Qm9vwnDqI1jW8dBFE5xi+Oz/LSEt/s1G5wUSjyTdmGD25sMmo/3yZ8UyDfGMn
      /UzjJ6p5xsculBrnnGc2idNnN1lmwGDCpcZksuECm0nCeQD9cB5S''')
    with self.subTest('traction'): self.assertAlmostEqual64(state['t'], '''
      eNrjPXH7pMbJJ+cZoGDyCYvLIFr7pJEBiOY+oW3GQCEAAGUgCg4=''')

  @testing.requires('matplotlib')
  def test_quadratic(self):
    cons, state = main(nelems=4, etype='square', btype='std', degree=2, poisson=.25)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons['u'], '''
      eNpjYCACNIxCfBAAg5xIAQ==''')
    with self.subTest('solution'): self.assertAlmostEqual64(state['u'], '''
      eNqFzT9LwlEUxvFLIj/FQQjLftGmtNj1nHvOHZqK8A0EvoKWwCEFQRRaWsJaazBbkja3AufEbGi711tT
      1Np/2oSUiO476Fk/X3iE+H9N/IQihPketGQbHnPnIEQbsvc9KLkivIyamLINlcaCagCo3XxWTVYySois
      Cu5A7Y8K6sD7m8lRHdP0BHGahak6lT++maptF6cvm6aMzdGhuaQ97NIYOrQMbbqVJ+S/aNV16MF2KWG9
      myU+wpADTPEaJPlZJlmIJC+6FF/bkCfey6bOx1jjGFZ5HSr8Ksu+qfCCq/LA1vjb+4654RYOOYED3oA+
      v8sr3/R53g24b4c89l4yMX2GgZ7DqN6EiP6VES1ERM+4qL6wgf7wvmX+AN5xajA=''')
    with self.subTest('traction'): self.assertAlmostEqual64(state['t'], '''
      eNpbejz8pNcpFdO5J26d5jy5y+DwCQYGzpNu5+eeUDPxOnXn1NLjK80YRgFeAAC0chL2''')

  @testing.requires('matplotlib')
  def test_poisson(self):
    cons, state = main(nelems=4, etype='square', btype='std', degree=1, poisson=.4)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons['u'], '''
      eNpjYMACGqgLASCRFAE=''')
    with self.subTest('solution'): self.assertAlmostEqual64(state['u'], '''
      eNpjYMAEOcZHje8byRhdN2JguG/05GyOsfWZkyYyJnNMzhl1mDAwzDExOnvS5MnpmaYmJp2mj4waTBkY
      Ok3lzs40PXPayMzRRMvsg5GaGQODlpnAWSOz/acBAbAecQ==''')
    with self.subTest('traction'): self.assertAlmostEqual64(state['t'], '''
      eNrbdFz7ROrJs8aHTjAwpJ40PrPp+FVzBioDANbTCtc=''')

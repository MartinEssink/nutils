#! /usr/bin/env python3
#
# In this script we solve Poisson's equation, :math:`u_{,kk} = 1`, using the
# fact that the solution of the strong form minimizes the functional
# :math:`\int_Ω \tfrac12 u_{,k} u_{,k} - u`. The domain :math:`Ω` is a unit
# square, discretized in a configurable number of elements, with the solution
# constrained to zero along the entire boundary.

from nutils import mesh, function, solver, export, cli

def main(nelems: int):
  '''
  Poisson's equation on a unit square.

  .. arguments::

     nelems [10]
       Number of elements along edge.
  '''

  domain, x = mesh.unitsquare(nelems, etype='square')
  basis = domain.basis('std', degree=1)
  u = basis @ function.Argument('udofs', shape=basis.shape)
  g = u.grad(x)
  J = function.J(x)
  cons = solver.optimize('udofs',
    domain.boundary.integral(u**2 * J, degree=2), droptol=1e-12)
  udofs = solver.optimize('udofs',
    domain.integral((g @ g / 2 - u) * J, degree=1), constrain=cons)
  bezier = domain.sample('bezier', 3)
  x, u = bezier.eval([x, u], udofs=udofs)
  export.triplot('u.png', x, u, tri=bezier.tri, hull=bezier.hull)

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. To
# keep with the default arguments simply run :sh:`python3 poisson.py`.

if __name__ == '__main__':
  cli.run(main)

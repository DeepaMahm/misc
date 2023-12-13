"""
Solving 1D concentration diffusion problem


References:
   Boundary condition:  https://github.com/lululxvi/deepxde/blob/b3921720188f0bf9183c3997dc60e5b26c17ae48/docs/demos/pinn_forward/eulerbeam.rst
   Initial condition: https://github.com/lululxvi/deepxde/blob/b3921720188f0bf9183c3997dc60e5b26c17ae48/docs/demos/pinn_forward/ode.2nd.rst#L11
"""
import deepxde as dde
import numpy as np


def pde(x, y):
    """
    :param x: 2-dimensional vector; x[:, 0:1] = x-coordinate; x[:, 1:] = t-coordinate
    :param y: network output; i.e., solution y(x,t)
    :return:
    """
    dc_t = dde.grad.jacobian(y, x, i=0, j=1)
    dc_xx = dde.grad.hessian(y, x, i=0, j=0)
    d = 1
    return (
        dc_t
        - d * dc_xx
    )


def boundary_l(x, on_boundary):
    """
    Because of rounding-off errors, dde.utils.isclose(x[0], 0)
    :param x:
    :param on_boundary: all boundary points
    :return:
    """
    return on_boundary and dde.utils.isclose(x[0], 0)  # définit le boundary x=0, left


def boundary_r(x, on_boundary):
    """
    Because of rounding-off errors, dde.utils.isclose(x[0], 0)
    :param x:
    :param on_boundary: all boundary points
    :return:
    """
    return on_boundary and dde.utils.isclose(x[0], 1)  # définit le boundary x=L, right


if __name__ == '__main__':
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # bc_l = dde.icbc.DirichletBC(geom, lambda x: 5, lambda _, on_boundary: boundary_l)
    # bc_r = dde.icbc.NeumannBC(geom, lambda x: 0, lambda _, on_boundary: boundary_r)

    bc_l = dde.icbc.DirichletBC(geomtime, lambda x: 5, boundary_l)
    bc_r = dde.icbc.NeumannBC(geom, lambda x: 0, boundary_r)

    ic = dde.icbc.IC(geomtime, lambda x: 1, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_l, bc_r, ic],
        num_domain=40,  # no. of training residual points sampled inside the domain
        num_boundary=20,  # no. of training points sampled on the boundary
        num_initial=10,  # initial residual points for initial conditions
        # solution=func, # reference solution to compute error of the dde solution, can be ignored if no exact soln
        num_test=10000,  # points for testing the residual
    )

    layer_size = [2] + [32] * 3 + [1]  # width 32, depth 3
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(iterations=1000)

    #dde.saveplot(losshistory, train_state, issave=True, isplot=True)

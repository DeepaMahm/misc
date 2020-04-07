import numpy as np

from gekko import GEKKO
from pprint import pprint
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def get_mmt():
    """
    M and M transpose required for differential equations
    :params: None
    :return: M transpose and M -- 2D arrays ~ matrices
    """
    # M^T
    MT = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, -1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, -1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, -1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, -1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, -1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, -1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, -1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, -1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    M = np.transpose(MT)
    return M, MT


def actual(phi, t):
    """
    Actual system/ Experimental measures
    :param  phi: 1D array
    :return: time course of variable phi -- 2D arrays ~ matrices
    """

    # spatial nodes
    ngrid = 10
    end = -1
    M, MT = get_mmt()
    D = 5000*np.ones(ngrid-1)
    A = MT@np.diag(D)@M
    A = A[1:ngrid-1]

    # differential equations
    dphi = np.zeros(ngrid)
    # first node
    dphi[0] = 0

    # interior nodes
    dphi[1:end] = -A@phi  # value at interior nodes

    # terminal node
    dphi[end] = D[end]*2*(phi[end-1] - phi[end])

    return dphi


def model(phi_hat, Dhat):
    """
    Model of the system / Measured
    :param phi_hat: 1D array
    :return: time course of variable phi
    """
    # spatial nodes
    ngrid = 10
    end = -1

    M, MT = get_mmt()

    A = MT@np.diag(Dhat)@M
    A = A[1:ngrid-1]

    # differential equations

    # first node
    m.Equation(phi_hat(0).dt() == 0)

    # interior nodes
    int_value = -A@phi_hat  # value at interior nodes
    m.Equations(phi_hat(i).dt() == int_value(i) for i in range(0, ngrid))

    # terminal node
    m.Equation(phi_hat(ngrid).dt() == Dhat[end]*2*(phi_hat(end-1) - phi_hat(end)))


if __name__ == '__main__':
    # ref: https://apmonitor.com/do/index.php/Main/PartialDifferentialEquations
    ngrid = 10  # spatial discretization
    end = -1

    # integrator settings (for ode solver)
    tf = 0.05
    nt = int(tf / 0.001) + 1
    tm = np.linspace(0, tf, nt)

    # ------------------------------------------------------------------------------------------------------------------
    # measurements
    # ref: https://www.youtube.com/watch?v=xOzjeBaNfgo
    # using odeint to solve the differential equations of the actual system
    # ------------------------------------------------------------------------------------------------------------------

    phi_0 = np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    phi = odeint(actual, phi_0, tm)

    # ------------------------------------------------------------------------------------------------------------------
    #  GEKKO model
    # ------------------------------------------------------------------------------------------------------------------
    m = GEKKO(remote=False)
    m.time = tm

    # ------------------------------------------------------------------------------------------------------------------
    # initialize phi_hat
    # ------------------------------------------------------------------------------------------------------------------

    phi_hat = [m.Var(value=phi_0[i]) for i in range(ngrid)]

    # ------------------------------------------------------------------------------------------------------------------
    # state variables
    # ------------------------------------------------------------------------------------------------------------------

    #phi_hat = m.CV(value=phi)
    #phi_hat.FSTATUS = 1  # fit to measurement phi obtained from 'def actual'

    # ------------------------------------------------------------------------------------------------------------------
    # parameters (/control variables to be optimized by GEKKO)
    # ref: http://apmonitor.com/do/index.php/Main/DynamicEstimation
    # def model
    # ------------------------------------------------------------------------------------------------------------------

    Dhat0 = 5000*np.ones(ngrid-1)
    Dhat = [m.FV(value=Dhat0[i]) for i in range(0, ngrid-1)]
    # Dhat.STATUS = 1  # adjustable parameter

    # ------------------------------------------------------------------------------------------------------------------
    # differential equations
    # ------------------------------------------------------------------------------------------------------------------

    M, MT = get_mmt()
    A = MT @ np.diag(Dhat) @ M
    A = A[1:ngrid - 1]

    # first node
    m.Equation(phi_hat[0].dt() == 0)

    # interior nodes
    int_value = -A @ phi_hat  # function value at interior nodes
    pprint(int_value.shape)
    m.Equations(phi_hat[i].dt() == int_value[i] for i in range(0, ngrid-2))

    # terminal node
    m.Equation(phi_hat[ngrid-1].dt() == Dhat[end] * 2 * (phi_hat[end-1] - phi_hat[end]))

    # ------------------------------------------------------------------------------------------------------------------
    # objective
    # ------------------------------------------------------------------------------------------------------------------
    # f = sum((phi(:) - phi_tilde(:)).^2);(MATLAB)
    # m.Minimize()

    # ------------------------------------------------------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------------------------------------------------------
    m.options.IMODE = 4  # simultaneous dynamic estimation
    m.options.NODES = 5  # collocation nodes
    m.options.EV_TYPE = 2  # squared-error :minimize model prediction to measurement
    m.solve()

    """
    #------------------------------------------------------------------------------------------------------------------
    #   Solving differential equation in GEKKO
    #------------------------------------------------------------------------------------------------------------------ 
    m.options.IMODE = 4  # simultaneous dynamic estimation
    m.solve(disp=True)
    """
    # plot results
    plt.figure()
    plt.figure()
    plt.plot(tm * 60, phi[:, :])
    plt.ylabel('phi')
    plt.xlabel('Time (s)')
    plt.show()

    plt.figure()
    for i in range(0, ngrid):
        plt.plot(tm*60, phi_hat[i].value)
    plt.ylabel('phi')
    plt.xlabel('Time (s)')
    plt.xlim([0, 3])
    plt.show()
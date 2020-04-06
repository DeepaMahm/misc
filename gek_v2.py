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


if __name__ == '__main__':
    # ref: https://apmonitor.com/do/index.php/Main/PartialDifferentialEquations
    ngrid = 10  # spatial discretization
    end = -1

    # integrator settings (for ode solver)
    tf = 0.5
    nt = int(tf / 0.01) + 1
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
    # initialize state variables: phi_hat
    # ref: https://apmonitor.com/do/uploads/Main/estimate_hiv.zip
    # ------------------------------------------------------------------------------------------------------------------

    phi_hat = [m.CV(value=phi_0[i]) for i in range(ngrid)]  # initialize phi_hat; variable to match with measurement

    for i in range(ngrid):
        phi_hat[i].FSTATUS = 1  # fit to measurement phi obtained from 'def actual'
        phi_hat[i].STATUS = 1  # build objective function to match measurement and prediction
        phi_hat[i].value = phi[:, i]

    # ------------------------------------------------------------------------------------------------------------------
    # parameters (/control parameters to be optimized while minimizing the cost function in GEKKO)
    # ref: http://apmonitor.com/do/index.php/Main/DynamicEstimation
    # ref: https://apmonitor.com/do/index.php/Main/EstimatorObjective
    # def model
    # ------------------------------------------------------------------------------------------------------------------
    #  Manually enter guesses for parameters
    Dhat0 = 5000*np.ones(ngrid-1)
    Dhat = [m.FV(value=Dhat0[i]) for i in range(0, ngrid-1)]
    for i in range(ngrid-1):
        Dhat[i].STATUS = 1  # Allow optimizer to fit these values

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
    m.Equations(phi_hat[i].dt() == int_value[i] for i in range(0, ngrid-2))

    # terminal node
    m.Equation(phi_hat[ngrid-1].dt() == Dhat[end] * 2 * (phi_hat[end-1] - phi_hat[end]))

    # ------------------------------------------------------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------------------------------------------------------
    m.options.IMODE = 5  # simultaneous dynamic estimation
    m.options.NODES = 5  # collocation nodes
    m.options.EV_TYPE = 2  # squared-error :minimize model prediction to measurement
    m.solve()

    """
    # plot results
    plt.figure()
    tm = m.time*60.0
    plt.plot(tm, phi_hat[9].value)
    plt.ylabel('phi')
    plt.xlabel('Time (s)')
    plt.xlim([0, 50])
    plt.show()
    """

'''Numerical recipes for HW6'''
import time
from numpy import float64 as flt, ndarray, ufunc, arange, array, empty_like, identity, ones, abs, concat, empty, exp, inf, log2, log10, nan, zeros
from numpy.linalg import inv, matrix_power as mp, norm
import matplotlib.pyplot as plt
from scipy.linalg import expm



def X_exact(t: float,  # time
      A: ndarray | float,  # nxn-matrix
      x_0: ndarray | float  # init. state: n-vector
      ) -> ndarray: #| float:
    '''exact slu. for linear difference equation'''
    #Exp = (expm if (type(A) is ndarray) else exp)
    #return Exp(A * t) @ x_0
    x_0 = (x_0 if (type(x_0) is ndarray) else array([x_0]))  # secure type/shape of x_0
    return expm(A * t) @ x_0  # works for both scalar & matrix, returns matrix



def P(h: float, Lambda: ndarray | float, method: str = 'fe') -> ndarray:
    '''returns update matrix P'''
    #Lambda = (array([Lambda]) if (type(Lambda) is float) else Lambda)
    Lambda = (Lambda if (type(Lambda) is ndarray) else array([[Lambda]]))  # secure type/shape of Lambda
    id = identity(len(Lambda))
    A = h * Lambda
    if method == 'be':
        #return mp(id - A, -1)
        return inv(id - A)
    elif method == 'tr':
        #return mp(id - A / 2, -1) @ (id + A / 2)
        return inv(id - A / 2) @ (id + A / 2)
    else:  # FE
        return id + A



def propagate(time_grid: ndarray, Lambda: ndarray | float, init: ndarray | float, mode: str = 'primal', prec: int = 1, method: str = 'tr') -> ndarray:
    '''generate approx. sequence for X, e, Psi, etc. for given time (step) grid'''
    len_t = len(time_grid)
    if type(init) is ndarray:  # secure type/shape of init
        len_i = len(init)
    else:
        len_i = 1
        init = array([init])
    #len_i = (len(init) if (type(init) is ndarray) else 1)
    #res = empty_like(time_grid)
    res = empty((len_t+1, len_i))
    primal_true = (mode == 'primal')
    if primal_true:
        res[0] = init
    else:
        res[-1] = init

    for n in range(len_t):
        m = (n if primal_true else -n-1)
        h = time_grid[m]

        if prec == 1:
            A = P(h, Lambda, method)
            #A_T = P(h, Lambda.T, method)
        elif prec == 2:
            A = P(h/2, Lambda, method)
            A @= A
            #A_T = P(h/2, Lambda.T, method)
            #A_T @= A_T
        else:
            A = mp(P(h/prec, Lambda, method), prec)
            #A_T = mp(P(h/prec, Lambda.T, method), prec)

        #res[m] = (A @ res[m-1] if primal_true else A.T @ res[m+1])
        if primal_true:
            res[m+1] = A @ res[m]
        else:  # dual
            res[m-1] = A.T @ res[m]
            #res[m-1] = A_T @ res[m]  # option for Lambda as general square matrix, makes difference?

    return res



def adapt(Lambda: ndarray | float, x_0: ndarray | float, g: ndarray | float, T: float = 1, #1 #2 #5 #10
          h_0: float = 1e-1, #p: int = 2,  # trap.
          TOL_range: int = 10, #5 #8 #10
          TOL_0: float = 1, #1 #2 #5 #8 #10 #20 #100
          TOL_ord: float = 2,
          s: int = 2, #S: int = 8, #4 #8  # for MSTZ-algorithm
          #legend: list[str] = ['adaptive', 'uniform'],
          legend: list[str] = ['MSTZ const. S', 'MSTZ adapt. S', 'Uniform Halving'],
          x_label: str = 'Neg. Log. Error Tolerance -log₂(TOL)', y_label: str = 'Log. Number of time steps log₂(N)'
          ) -> tuple[ndarray, ndarray, list[str], str, str]:
    '''Ex.d: returns numbers of time steps N for (several) adaptivity methods given range TOL'''
    #gamma = 0.5 / (p + 1)
    TOLs = TOL_0 * TOL_ord**-arange(TOL_range, dtype=flt)  # range of error tolerance for which to compute and plot Ns
    Ns = []  # collect #time steps over TOLs
    N_0 = int(T / h_0)
    #Exp = (expm if (type(Lambda) is ndarray) else exp)

    for rnd, TOL in enumerate(TOLs):
        #print(f"{k}, ", end="")  # print round
        print(f"\n\nround {rnd}:")

        #print("adaptive: ", end="")
        print("MSTZ const.: ", end="")
        N_const = MSTZ(False, TOL, h_0, s, Lambda, x_0, g, N_0)
        print("\nMSTZ adapt.: ", end="")
        N_adapt = MSTZ(True, TOL, h_0, s, Lambda, x_0, g, N_0)
        print("\nuniform: ", end="")
        N_unif = uniform(TOL, h_0, Lambda, x_0, g, N_0)

        Ns.append([N_const, N_adapt, N_unif])

    return -log2(TOLs), log2(array(Ns)), legend, x_label, y_label



def uniform(TOL: float, h_0: float,  # main
            Lambda: ndarray | float, x_0: ndarray | float, g: ndarray | float, N: int) -> int:  # given
    '''return number of time steps for uniform halfing method for given TOL'''
    k = 0
    h = h_0
    time_grid = h_0 * ones(N, dtype=flt)
    while(True):  # for unif. time step method
        print(f"{k}, ", end="")
        #len_t = len(time_grid_2)
        #new_grid = [] #[0.]
        r = err_ind(time_grid, TOL, N, Lambda, x_0, g)
        r_approx = TOL / N
        #if r.max(axis=0) <= S * r_approx:  # overall stopping condition for unif. method
        if r.max(axis=0) <= r_approx:  # w/o S
            break
        h /= 2  # factor-2 updates
        N *= 2
        time_grid = h * ones(N, dtype=flt)  # double unif. time grid precision
        k += 1
    return N



def MSTZ(adapt_S: bool, TOL: float, h_0: float, s: int,  # main
         Lambda: ndarray | float, x_0: ndarray | float, g: ndarray | float, N: int) -> int:  # given
    '''return number of time steps for adaptive MSTZ method for given TOL, choose mode via adapt_S'''
    k = 0
    loop = True
    r_old = None
    time_grid = h_0 * ones(N, dtype=flt)
    while(loop):  # for adaptive MSTZ-style method
        print(f"{k}, ", end="")
        loop = False  # use w/o MSTZ-overall-condition
        len_t = len(time_grid)
        new_grid = [] #[0.]
        r = err_ind(time_grid, TOL, len_t, Lambda, x_0, g)
        r_approx = TOL / N
        if (r_old is not None) and adapt_S:
            S = 2 * s * r.max(axis=0) / r_old.min(axis=0)  # adapt c-hat
        else:
            S = 4 * s  # init./const.
        if r.max(axis=0) <= S * r_approx:  # overall stopping condition for MSTZ
            break
        for n in range(len_t):
            if r[n] > s * r_approx:
            #if r[n] >  r_approx:  # simple version w/o MSTZ
                new_step = time_grid[n] / 2
                new_grid += [new_step, new_step]  # add 2 half-steps to new_grid
                loop = True
            else:
                new_grid.append(time_grid[n])  # same step
        r_old = r
        time_grid = array(new_grid)  # update time_grid
        N = len(time_grid)  # update #time steps
        k += 1
    return N



def err_ind(time_grid: ndarray, TOL: float, len_t: int, Lambda: ndarray | float, x_0: ndarray | float, g: ndarray | float, p: int = 2 # trap.
            ) -> ndarray:
    '''return error indicator for given time grid, etc.'''
    gamma = 0.5 / (p + 1)  # = 1/6
    shape = (1, len_t)
    #len_t = len(time_grid)
    #new_grid = [] #[0.]
    times = zeros(len_t+1, dtype=flt)
    for n in range(len_t):  # infer times from time_grid
        times[n+1] = times[n] + time_grid[n]
    X_1 = propagate(time_grid, Lambda, x_0, 'primal')  # simple approx. of X, gets too long for too many rounds
    X = array([X_exact(t, Lambda, x_0) for t in times])  # exact X over times
    error = X - X_1  # exact error
    Psi = propagate(time_grid, Lambda, g, 'dual')  # dual: approx. 1st variation from dual problem
    r = abs((error * Psi).sum(axis=-1))  # prelim. error indicator, scalar prod. version
    #r = norm(error, axis=-1) * norm(Psi, axis=-1) # prelim. error indicator, upper bound via norms
    r = concat([r[1:].reshape(shape), TOL**gamma * time_grid.reshape(shape)**(p+1)], axis=0).max(axis=0)  # apply regularization/floor term
    return r



def Errors(T: float = 10, TOL: float = 1e-3, h_0: float = 1e-1, k_max: int = 20, #14 #15 #19 #20
           pos: int = 2, log_N: ufunc = log10, log_err: ufunc = log10,
           #control: float = 1e2, # control numerical instability of Lambda-multiplications
           x_0: ndarray = array([1, 2, 3], dtype=flt), #.transpose()
           Lambda: ndarray = array([[-4000, 4003, -1994],
                                   [3997, -4000, 2006],
                                   [-2006, 1994, -1000]], dtype=flt) / 9,
            legend: list[str] = [#'Forward Euler',
               'Trapezoidal',
               #'F.E. Richardson refine.',
               'Trap. Richardson refine.',
               #'F.E. Richardson approx.',
               'Trap. Richardson approx.',
               'TOL'],
            x_label: str = 'Log. Number of time steps log₁₀(N)', y_label:str = 'Log. Final Errors log₁₀(|e(T)|)'
            ) -> tuple[ndarray, ndarray, list[str], str, str]: #| None:
    '''numerical recipes for Ex.c'''
    #k: int = 0  # time step scale
    #Delta_ctrl = Lambda / control  # apply control factor
    id = identity(3)
    Ns, errors = [], []
    #round: int = 1
    h = h_0
    N = int(T / h_0)
    N_half = int(N / 2)

    #while(True):
    for k in range(k_max):
        #h = h_0 * 2**-k  # time step
        #N = int(T / h)  # number of time steps
        #N_half = int(N / 2)
        #h_2 = 2 * h  # 2x time step
        h_2 = h / 2  # 1/2 time step
        P_fe = P(h, Lambda, 'fe')  # update matrix for Forward Euler
        P_tr =  P(h, Lambda, 'tr') # update matrix for Trapezoidal
        P_ri_fe = P(h_2, Lambda, 'fe')
        P_ri_tr = P(h_2, Lambda, 'tr')
        #P_fe /= control  # apply control factor
        #P_tr /= control
        X_fe, X_tr, X_ri_fe, X_ri_tr = x_0, x_0, x_0, x_0  # initial condition
        #k += 1  # update time step scale
        print(f"{k}, ", end="")
        #round += 1

        #for n in range(1, N_half+1):  # 1st half run
            #X_fe = P_fe @ P_fe @ X_fe  # step-wise FE updates, half-step
            #X_tr = P_tr @ P_tr @ X_tr  # step-wise Trap. updates, half-step
            # step-wise updates for half-precision (2x time step) states:
            #X_ri_fe = P_ri_fe @ X_ri_fe
            #X_ri_tr = P_ri_tr @ X_ri_tr
        #X_fe, X_tr = mp(P_fe, N_half) @ X_fe, mp(P_tr, N_half) @ X_tr  # direct computation
        #X_ri_fe, X_ri_tr = X_fe, X_tr  # half point for Richardson

        #for n in range(N_half+1, N+1):  # 2nd half run
        for n in range(1, N+1):  # full run
            X_fe = P_fe @ X_fe  # step-wise FE updates
            X_tr = P_tr @ X_tr  # step-wise Trap. updates
            # step-wise updates for double-precision (1/2 time step) states:
            X_ri_fe = P_ri_fe @ P_ri_fe @ X_ri_fe
            X_ri_tr = P_ri_tr @ P_ri_tr @ X_ri_tr
        #X_fe, X_tr = mp(P_fe, N_half) @ X_fe, mp(P_tr, N_half) @ X_tr  # direct computation

        X_ri_fe = 2 * X_ri_fe - X_fe # Richardson for Forward Euler (p=1)
        X_ri_tr = (4 * X_ri_tr - X_tr) / 3  # Richardson for Trapezoidal (p=2)
        #X_ri_tr = (8 * X_ri_tr - X_tr) / 7  # Richardson for Trapezoidal for p+1
        """#
        error = X_exact(T, Lambda, x_0)[pos] - array([#X_fe[pos],
                                                   X_tr[pos],
                                                   #X_ri_fe[pos],
                                                   X_ri_tr[pos]])  # array of errors of approx.s wrt. exact slu. at time T
        """#
        X, X_fe, X_tr = X_exact(T, Lambda, x_0)[pos], X_fe[pos], X_tr[pos]  # from vectors to last elem.
        error = array([#X - X_fe,
                       X - X_tr,
                       #X - X_ri_fe[pos],
                       X - X_ri_tr[pos],  # 'exact' error for Rich.
                       #X_fe - X_ri_fe[pos],
                       X_tr - X_ri_tr[pos]  # 'approx.' error for Rich.
                       ])
        #error = X_exact(T, Lambda, x_0)[pos]
        #error -= array([X_fe[pos], X_tr[pos], X_ri_fe[pos], X_ri_tr[pos]])
        error = abs(error)

        #err_max = abs(error).max()
        thresh = error.max()  # break after last to drop below TOL
        #thresh = error.min()  # break after 1st to drop below TOL
        if (thresh < TOL) or (thresh == inf) or (nan in error):  # break conditions
            break
        errors.append(error)  # collect errors
        Ns.append(N)  # collect numbers of time steps
        # scale updates:
        N *= 2
        N_half *= 2
        h /= 2

    #length = len(errors)
    length = len(Ns)
    Ns = array(Ns, dtype=flt)
    if length > 1:
        print("\n", end="")
        #return log2(Ns), concat([log_err(array(errors)) + log_err(control) * Ns.reshape((length, 1)),  # revert control factor on log-lvl
        #                         log_err(TOL) * ones((length, 1))], axis=-1)
        return log_N(Ns), log_err(concat([array(errors), TOL * ones((length, 1))], axis=-1)), legend, x_label, y_label  # w/o control
    else:
        warn_txt = "\nNs & errors not long enough! Will return default."
        #ValueError(warn_txt)
        #Warning(warn_txt)
        print(warn_txt)
        return array([1, 2]), ones((2, len(legend))), legend, x_label, y_label



def plot(x: ndarray, y: ndarray, legend: list[str], x_label: str, y_label: str) -> None:
    '''plot local error of X_3(10) against #timesteps N for various numerical methods'''
    fig, ax = plt.subplots(figsize = (16,9))
    for col in range(y.shape[-1]):
        ax.plot(x, y[:, col], linewidth=2.0)
    #ax.set_title('', fontsize=20, pad=12.0)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(x_label, fontsize=20, labelpad=8.0)
    ax.set_ylabel(y_label, fontsize=20, labelpad=12.0)
    #ax.set_xbound(lower=0.0, upper=periods)
    #ax.set_ybound(lower=0.0)
    ax.grid(which='major', axis='both')
    #ax.spines['bottom'].set_position(('axes', 0.0))
    ax.legend(legend, fontsize=13, title_fontsize=22, markerscale=6.0)  #title=''
    plt.show()



if __name__ == "__main__":
    from timeit import default_timer
    t = default_timer()
    Lambda = 4.0 #2**0 #1.0 #2.0 #4.0
    #x, y, legend, x_label, y_label = Errors()
    x, y, legend, x_label, y_label = adapt(Lambda, x_0=1., g=1.)
    print(f"\nCompute Time: {(default_timer() - t)} secs\n---")
    plot(x, y, legend, x_label, y_label)

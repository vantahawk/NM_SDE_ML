import matplotlib.pyplot as plt
from numpy import ndarray, arange, array, identity, linspace, ones, zeros, abs, cos, exp, heaviside, log, log2, log10, maximum, nan, pi
from numpy.random import default_rng
from scipy.special import ndtr, ndtri



def plot(x: ndarray, data: list[ndarray], legend: list[str], axes: tuple[str, str], title: str = '') -> None:
    '''for 1a, diff-part: plot simulated df(0,s)/ds against its analytic slu. over M steps on range (0, s_max]'''
    fig, ax = plt.subplots(figsize = (16,9))
    axX, axY = axes
    #legend = ['num-direct', 'num-num'
    #          #, 'analytic'
    #          ]
    for y in data:
        ax.plot(x, y, linewidth=2.0)
        #legend.append(name)
    ax.set_title(title, fontsize=20, pad=12.0)
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_xlabel('log₁₀(\u03B1)', fontsize=20, labelpad=8.0)
    #ax.set_xlabel('Initial Stock Price s', fontsize=20, labelpad=8.0)
    ax.set_xlabel(axX, fontsize=20, labelpad=8.0)
    #ax.set_ylabel('Lower bound on \u0394t(\u03B1)', fontsize=20, labelpad=12.0)
    #ax.set_ylabel('Derivative of Option Price df(0,s)/ds', fontsize=20, labelpad=12.0)
    ax.set_ylabel(axY, fontsize=20, labelpad=12.0)
    #ax.set_xbound(lower=0.0)
    #ax.set_ybound(lower=0.0)
    ax.grid(which='major', axis='both')
    #ax.spines['bottom'].set_position(('axes', 0.0))
    ax.legend(legend, fontsize=13, title_fontsize=22, markerscale=6.0)  #title=''
    plt.show()



def phi(x: float, deriv: bool = False) -> float:
    '''PDF of standard normal dist.'''
    res = exp(-x**2 / 2) / (2 * pi)**0.5
    if deriv:
        return -x * res
    else:
        return res



def option(S: ndarray | float, K: float,
           M: int,
           return_res: int = 1#,
           #relu: bool = True
           ) -> float | ndarray:
    '''returns option-function term w/ MC-approx. of expectation over n processes'''
    #y = x - K
    #return y if y > 0 else 0
    #return maximum(S - K * ones(n, dtype=float), zeros(n, dtype=float)).mean
    #return maximum(S - K, 0).mean(dtype=float)  # use broadcasting of K & 0 to S-shape
    res = maximum(S - K, 0)
    """
    if relu:
        res = maximum(S - K, 0)
    else:
        res = heaviside(S - K, 0)
    """
    #n = len(S)
    #res = maximum(S - K * ones(n, dtype=float), zeros(n, dtype=float))
    if return_res == 1:  # f-estimator = sample mean
        return res.mean(dtype=float)
    if return_res == 2:
        return (res.mean(dtype=float, axis=1)).std(axis=0, dtype=float, out=None, ddof=1, keepdims=False, where=True)  # for std of f-estimator for CLT using MC^2
    if return_res == 3:
        return res.std(axis=None, dtype=float, out=None, ddof=1, keepdims=False, where=True) / M**0.5  # simple std of f-estimator for CLT w/ additional sqrt(M)-decay
        #return res.var(axis=None, dtype=float, out=None, ddof=1, keepdims=False, where=True)**0.5
    else:
        return res

"""
def MC1(s: float = 35, num: bool = False, K: float = 35, r: float = 0.04, sigma: float = 0.2, T: float = 0.5, n: int = 100, N: int = 1000) -> float:
    '''Monte-Carlo approx. of f(0,s) (1D, Ex.1a), i.e. expected option price at time 0 w/ strike time T, simulate n iid. Itô-processes, take sample average over option-function terms at the end, multiply w/ exp. drift term & return, choose betw. direct vs. numerical simulation via Forward-Euler over N time steps '''
    rng = default_rng(seed=None)

    if num:  # forward Euler
        delta_t = T / N  # uniform time step
        drift = r * delta_t  # deterministic drift increment
        diffusion = sigma * delta_t**0.5  # scaling factor from SN-dist. to diffusion increment

        S = s * ones(n, dtype=float)
        for t in range(N):
            S += S * (drift + diffusion * rng.standard_normal(size=n, dtype=float))
    else:
        S = s * exp((r - sigma**2 / 2) * T + sigma * T**0.5 * rng.standard_normal(size=n, dtype=float))

    return exp(-r * T) * option(S, K)
"""


def MC1(M: int,
        return_res: int = 1,
        relu: bool = True,
        given_wiener: float | ndarray = 0, #nan,
        w_new: bool = True,
        s: float = 35, K: float = 35, r: float = 0.04, sigma: float = 0.2, T: float = 0.5):
    '''Monte-Carlo approx. of f(0,s) (1D, Ex.1a), i.e. expected option price at time 0 w/ strike time T, simulate n iid. Itô-processes (at time T), take sample average over option-function terms at the end, multiply w/ exp. drift term & return, choose between using new Wiener variable or given value'''

    #if given_wiener == nan:  # default
    if w_new:  # default
        if return_res == 2:
            size = (M, M)  # for std of f-estimator for CLT
        else:
            size = M
        rng = default_rng(seed=None)
        X_W = rng.standard_normal(size=size, dtype=float)  # draw new iid. standard gaussian r.v. for Wiener process
    else:
        X_W = given_wiener  # use given value for Wiener r.v.

    #drift1 = r * T
    #drift2 = -T * sigma**2 / 2
    #diffusion = sigma * T**0.5 * X_W
    S = s * exp((r - sigma**2 / 2) * T + sigma * T**0.5 * X_W)
    #S = s * exp(drift1 + drift2 + diffusion)
    if relu:
        return exp(-r * T) * option(S, K, M, return_res)
    else:
        return exp(-T * sigma**2 / 2) * (exp(sigma * T**0.5 * X_W) * heaviside(S - K, 0)).mean()



def f(ord: int = 0, t: float = 0, s: float = 35, K: float = 35, r: float = 0.04, sigma: float = 0.2, T: float = 0.5) -> float:
    '''[ord]-s-deriv. of f(t,s), i.e. exact slu. of BS equation'''
    a = sigma * (T - t)**0.5
    b = K * exp(r * (t - T))
    d_1 = (log(s / K) + (r + sigma**2 / 2) * (T - t)) / a
    d_2 = d_1 - a
    if ord == 0:
        res = s * ndtr(d_1) - b * ndtr(d_2)
    if ord == 1:
        res = ndtr(d_1)+ (phi(d_1) - b * phi(d_2) / s) / a
    return res  # type: ignore


"""
def MCdiff(delta_s: float, num: bool = False, s_max: float = 100, M: int = 100) -> tuple[ndarray, ndarray]:
    '''returns df(0,s)/ds over M steps on range (0, s_max], approx.d on the basis of successive forward diff-coeff.s wrt. MC-approx. in MC1d()'''
    steps = s_max * arange(1, M+1, dtype=float) / M
    diffs = array([(MC1(s + delta_s, num) - MC1(s, num)) / delta_s for s in steps])
    return steps, diffs  #(MC1(steps + delta_s) - MC1(steps)) / delta_s
"""

"""
def MCdiff(delta_s: float, same_wiener: bool = False, s_max: float = 100) -> tuple[ndarray, ndarray]:
    '''returns df(0,s)/ds over M steps on range (0, s_max], approx.d on the basis of successive forward diff-coeff.s wrt. MC-approx. in MC1d()'''
    steps = s_max * arange(1, n+1, dtype=float) / n
    diffs = array([(MC1(s + delta_s, num) - MC1(s, num)) / delta_s for s in steps])
    return steps, diffs  #(MC1(steps + delta_s) - MC1(steps)) / delta_s
"""


def MCdiff(delta_s: float, M: int, delta: int,
           #new_w: bool = True,
           s: float = 35) -> float:
    '''only at s=35'''
    X_W = 0
    #if not new_w:
    #    rng = default_rng(seed=None)
    #    X_W = rng.standard_normal(size=M, dtype=float)

    if delta == 1:  # exact 1st deriv. via BS
        return f(1)
    if delta == 2:  # exact delta term via BS
        return (f(0,0,s + delta_s) - f(0)) / delta_s
    if delta == 3:  # delta term of f-estimator w/ *indep.* Wiener terms
        return (MC1(M, 1, True, X_W, True, s + delta_s) - MC1(M, 1, True, X_W, True, s)) / delta_s
    if delta == 4:  # delta term of f-estimator w/ *same* Wiener term
        rng = default_rng(seed=None)
        X_W = rng.standard_normal(size=M, dtype=float)
        return (MC1(M, 1, True, X_W, False, s + delta_s) - MC1(M, 1, True, X_W, False, s)) / delta_s
    if delta == 5:  # CLT-variance term for stat. error w/ *indep.* Wiener terms
        return (MC1(M, 0, True, X_W, True, s + delta_s) - MC1(M, 0, True, X_W, True, s)).std(axis=None, dtype=float, ddof=1) / (delta_s * M**0.5)
        #return (MC1(M, 0, True, X_W, True, s + delta_s) - MC1(M, 0, True, X_W, True, s)).std(axis=None, dtype=float, ddof=1) / M**0.5
    if delta == 6:  # CLT-variance term for stat. error w/ *same* Wiener term
        rng = default_rng(seed=None)
        X_W = rng.standard_normal(size=M, dtype=float)
        return (MC1(M, 0, True, X_W, False, s + delta_s) - MC1(M, 0, True, X_W, False, s)).std(axis=None, dtype=float, ddof=1) / (delta_s * M**0.5)
        #return (MC1(M, 0, True, X_W, False, s + delta_s) - MC1(M, 0, True, X_W, False, s)).std(axis=None, dtype=float, ddof=1) / M**0.5
    else:  # 1st deriv. of f-estimator
        return MC1(M, 1, False, X_W, True, s)


"""
def MCd(s: float = 40, d: int = 10, K: float = 40, r: float = 0.04, T: float = 0.5, n: int = 100,
        #N: int = 1000
        ) -> float:
    '''Monte-Carlo approx. of f(0,s) (d-D, Ex.1b), i.e. expected _average_ option price at time 0 w/ strike time T, simulate n iid. instances of d coupled Itô-processes, take sample average over option-function terms at the end, multiply w/ exp. drift term & return, use random (non-diag.) diffusion matrix'''
    rng = default_rng(seed=None)
    Sigma = rng.standard_normal(size=(d, d), dtype=float)  # most likely non-diagonal diffusion matrix
    W = T**0.5 * rng.standard_normal(size=(d, n), dtype=float)  #
    S = s * exp(r * T) * exp(-T * (Sigma**2).sum(axis=1) / 2) @ exp(Sigma @ W) / d  # direct simulation, simplified representation to reduce broadcasting & redundant computation
    return exp(-r * T) * option(S, K)
"""

"""
def MC1_data(x: ndarray | list[int]) -> ndarray| list[int]:
    '''return value array for MC1(x)'''
    return array([MC1(M) for M in x])
    #return [MC1(M) for M in x]
"""

"""
def y_data(x: ndarray, func: callable) -> ndarray:
    '''return value range of func over x'''
    return array([func(M) for M in x])
"""


delta_s = 0.1
CI = 0.95  # confidence interval
#print(MC1())
#plot(([MCdiff(delta_s), MCdiff(delta_s, True)], 'Initial Stock Price s', 'Derivative of Option Price df(0,s)/ds'))
start_M = 3 #0 #1 #3 #7
end_M = 20 #17 #20
M_log = array(range(start_M, end_M))
#M_range = array([2**k for k in range(start, end)])
M_range = 2**M_log
#M_range = [2**k for k in range(start, end)]
MC1_data = array([MC1(M) for M in M_range])
f_data = f() * ones(end_M-start_M, dtype=float)
#f_error = abs(MC1_data - f_data)
f_error = log2(abs(MC1_data - f_data))
#f_clt = ndtri((CI + 1) / 2) * array([MC1(M, True) for M in M_range])
C_alpha = ndtri((CI + 1) / 2)
f_clt = log2(C_alpha * array([MC1(M, 3) for M in M_range]))
#f_clt_anal = C_alpha *

#MC1_data(M_range),
#y_data(M_range, MC1),

# Task 1
#plot(M_log, [MC1_data, f_data], ['MC approximation', 'analytic result'], ('Log. Number of Samples log\u2082(M)', 'Expected Option Price f(t=0, s=35)'))

# Task 2
#plot(M_log, [f_error, f_clt], ['actual error', 'estimated CLT bound'], ('Log. Number of Samples log\u2082(M)', 'log\u2082 of Error of Option Price for t=0, s=35'))

# Task 3
MM = 2**(end_M-1) #10 #15 #19
hh = 2**-4.7 #-0.77 #5 #10 #15 #20 #30
start_h = 0 #0 #0.6 #4
end_h = 17 #6 #17 #1.8 #2 #3 #7 #17 #20 #21 #22 #25 #30
n = 20
h_log = array(range(start_h, end_h), dtype=float)
#h_log = linspace(0.76, 0.775, n, dtype=float)
#h_log = linspace(start_h, end_h, n, dtype=float)
h_range = 2**-h_log
# exact deriv.
diff1_data_M = f(1) * ones(end_M-start_M, dtype=float)
diff1_data_h = f(1) * ones(end_h-start_h, dtype=float)
#diff1_data_h = f(1) * ones(n, dtype=float)
# Delta term
diff2_data_M = array([MCdiff(hh, M , 2) for M in M_range])
diff2_data_h = array([MCdiff(h, MM , 2) for h in h_range])
# MC-estimator of Delta term
diff3_data_M = array([MCdiff(hh, M , 3) for M in M_range])
diff3_data_h = array([MCdiff(h, MM , 3) for h in h_range])
diff4_data_M = array([MCdiff(hh, M , 4) for M in M_range])
diff4_data_h = array([MCdiff(h, MM , 4) for h in h_range])
# CLT bounds of MC-estimator of Delta term
diff5_data_M = C_alpha * array([MCdiff(hh, M , 5) for M in M_range])
diff5_data_h = C_alpha * array([MCdiff(h, MM , 5) for h in h_range])
diff6_data_M = C_alpha * array([MCdiff(hh, M , 6) for M in M_range])
diff6_data_h = C_alpha * array([MCdiff(h, MM , 6) for h in h_range])

# plot data
"""# 1(&5))
error_disc_h = log2(abs(diff2_data_h - diff1_data_h))
#error_stat1_h = log2(abs(diff3_data_h - diff2_data_h))
#error_stat2_h = log2(abs(diff4_data_h - diff2_data_h))
#error_tot1_h = log2(abs(diff3_data_h - diff1_data_h))
#error_tot2_h = log2(abs(diff4_data_h - diff1_data_h))
error_clt1_h = log2(diff5_data_h)
error_clt2_h = log2(diff6_data_h)
plot(h_log, [error_disc_h,
             #error_stat1_h, error_stat2_h, error_tot1_h, error_tot2_h,
             error_clt1_h, error_clt2_h
             ],
     ['discrete error',
      #'stat. error w/ indep. samples', 'stat. error w/ same sample', 'total error w/ indep. samples', 'total error w/ same sample',
      'CLT bound w/ indep. samples', 'CLT bound w/ same sample'
      ],
     ('Neg. Log. of s-Increment -log\u2082(h)', 'log\u2082 of Error of Deltas \u03b5(h)'),
     f'1) Error of \u0394-terms over h for t=0, s=35, M={MM}')
"""
"""# 2)
plot(M_log, [diff1_data_M, diff2_data_M,
             #diff3_data_M,
             diff4_data_M],
     ['derivative', 'finite',
      #'MC-estimator w/ indep. samples',
      'MC-estimator w/ same sample'],
     ('Log. Number of Samples log\u2082(M)', 'Deltas \u0394(M)'),
     f'2) \u0394-terms over M for t=0, s=35, h={hh}')
"""
"""# 3)
error_stat1_M = log2(abs(diff3_data_M - diff2_data_M))
error_stat2_M = log2(abs(diff4_data_M - diff2_data_M))
error_clt1_M = log2(diff5_data_M)
error_clt2_M = log2(diff6_data_M)
plot(M_log, [error_stat1_M, error_stat2_M, error_clt1_M, error_clt2_M],
     ['stat. error w/ indep. samples', 'stat. error w/ same sample', 'CLT bound w/ indep. samples', 'CLT bound w/ same sample'],
     ('Log. Number of Samples log\u2082(M)', 'log\u2082 of Error of Deltas \u03b5(h)'),
     f'3) Error of \u0394-terms over M for t=0, s=35, h={hh}')
"""
"""# 4)
plot(h_log, [diff1_data_h, diff2_data_h,
             #diff3_data_h,
             diff4_data_h],
     ['derivative', 'finite',
      #'MC-estimator w/ indep. samples',
      'MC-estimator w/ same sample'],
     ('Neg. Log. of s-Increment -log\u2082(h)', 'Deltas \u0394(h)'),
     f'4) \u0394-terms over h for t=0, s=35, M={MM}')
"""
# 5)
error_disc_h = abs(diff2_data_h - diff1_data_h)
error_stat1_h = log2(abs(diff3_data_h - diff2_data_h))
error_stat2_h = log2(abs(diff4_data_h - diff2_data_h))
#error_tot1_h = log2(abs(diff3_data_h - diff1_data_h))
#error_tot2_h = log2(abs(diff4_data_h - diff1_data_h))
error_tot1_h = log2(diff5_data_h + error_disc_h)
error_tot2_h = log2(diff6_data_h + error_disc_h)
plot(h_log, [error_stat1_h, error_stat2_h, error_tot1_h, error_tot2_h],
     ['stat. error w/ indep. samples', 'stat. error w/ same sample', 'total error bound w/ indep. samples', 'total error bound w/ same sample'],
     ('Neg. Log. of s-Increment -log\u2082(h)', 'log\u2082 of Error of Deltas \u03b5(h)'),
     f'5) Error of \u0394-terms over h for t=0, s=35, M={MM}')

'''Plots for Ex.2c of HW3'''
import matplotlib.pyplot as plt
from numpy import ndarray, identity, linspace, zeros, cos, exp, log10#, float as float
from numpy.random import default_rng



def plot(data: list[tuple[ndarray, ndarray]], elems:  list[int]) -> None:
    '''plot num. sample values for Var(e(t)) for diff. N'''
    fig, ax = plt.subplots(figsize = (16,9))
    legend = [f'N = {N}' for N in elems] + ['upper bound']
    for x, y in data:
        ax.plot(x, y, linewidth=2.0)
    #ax.set_title('', fontsize=20, pad=12.0)
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_xlabel('log₁₀(\u03B1)', fontsize=20, labelpad=8.0)
    ax.set_xlabel('Time t', fontsize=20, labelpad=8.0)
    #ax.set_ylabel('Lower bound on \u0394t(\u03B1)', fontsize=20, labelpad=12.0)
    ax.set_ylabel('Logarithm of Variance of Difference log₁₀(Var(e(t)))', fontsize=20, labelpad=12.0)
    ax.set_xbound(lower=0.0)
    #ax.set_ybound(lower=0.0)
    ax.grid(which='major', axis='both')
    #ax.spines['bottom'].set_position(('axes', 0.0))
    ax.legend(legend, fontsize=13, title_fontsize=22, markerscale=6.0)  #title=''
    plt.show()



def var_diff(N: int = 10, fe_mode: bool = True, M: int = 1000, T: float = 6, b: float = 0.1) -> tuple[ndarray, ndarray]:
    '''recursively produces forward euler values for diff. e(t)=X(t)-Z(t) over M iid. processes w/ given param.s and returns time-spanning array of sample variances of e(t)'''
    times = linspace(0, T, num=N+1, endpoint=True, retstep=False, dtype=float, axis=0, device=None)  # uniform time range based on given N

    if fe_mode:  # forward-euler mode: compute variances numerically
        # subparam.s:
        delta_t = T / N  # uniform time step
        wiener_scale = b * delta_t**0.5  # scaling factor from SN-dist. to wiener increment
        #mean = zeros(M, dtype=float)  # mean for multivar. normal dist.
        #cov = delta_t * identity(M, dtype=float)  # covar. mat. for multivar. normal dist.
        x_0 = zeros(M, dtype=float)
        #x_0 = mean  # start value = 0
        z, x = x_0, x_0  # start values
        #e = mean
        #times = linspace(0, T, num=N, endpoint=True, retstep=False, dtype=float, axis=0, device=None)  # uniform time range based on given N
        var_e = zeros(N+1, dtype=float)  # empty sample estimator for variances of e(t) over time
        rng = default_rng(seed=None)

        # forward euler recursion w/ a(x)=cos(x):
        for t in range(1, N+1):
            z = z + cos(z) * delta_t
            #x = x + cos(x) * delta_t + b * rng.multivariate_normal(mean, cov, size=None, check_valid='ignore', tol=1e-8, method='svd')
            x = x + cos(x) * delta_t + wiener_scale * rng.standard_normal(size=M, dtype=float)
            #e = x - z
            var_e[t] = (x - z).var(axis=None, dtype=float, out=None, ddof=1, keepdims=False, where=True)  # fill sample estimator for Var(e(t)) w/ Bessel correction

    else:  # compute upper bound w/ closed form result from Ex.2b
        #times = linspace(0, T, num=1e3, endpoint=True, retstep=False, dtype=float, axis=0, device=None)  # high-res. uniform time range
        var_e = 2 * T * (b**2) * exp(2 * T * times, dtype=float)  # C_a = 1 for a(x)=cos(x)

    #return times, var_e  # return tuple of time range & resp. sample variances of e(t)
    return times, log10(var_e)  # return log10 of variances
    #return linspace(0, T, num=N+1, endpoint=True, retstep=False, dtype=float, axis=0, device=None), var_e



t_res = [10, 20, 40]  # time range resolutions
#t_res = [10 * 2**k for k in range(8)]
plot_stack = [var_diff(N) for N in t_res] + [var_diff(1000, False)]  # list of times-values tuples for plotting, 1st numerical results then upper bound
plot(plot_stack, t_res)

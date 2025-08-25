#import numpy as np
from numpy import log, log10, pi
import matplotlib.pyplot as plt



def delta_t(alpha: float, T: float = 1.0, sigma: float = 0.7) -> float:
    '''closed-form slu. to Ex.2c from modified assumption (ii)'''
    return -(log(-(2 * pi)**0.5 * log(1 - alpha) / (T * sigma**2)) * 2 * sigma**2)**-1



def plot(x: list[int] | list[float], y: list[float]) -> None:
    '''plot delta_t on logarithmic scale'''
    fig, ax = plt.subplots(figsize = (16,9))
    #legend = []
    ax.plot(x,y, linewidth=2.0)
    #ax.set_title('', fontsize=20, pad=12.0)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('log₁₀(\u03B1)', fontsize=20, labelpad=8.0)
    ax.set_ylabel('Lower bound on \u0394t(\u03B1)', fontsize=20, labelpad=12.0)
    #ax.set_xbound(lower=0.0, upper=periods)
    ax.set_ybound(lower=0.0)
    ax.grid(which='major', axis='both')
    #ax.spines['bottom'].set_position(('axes', 0.0))
    #ax.legend(legend, fontsize=13, title_fontsize=22, markerscale=6.0)  #title=''
    plt.show()



#alphas = [10**-k for k in range(3, 9)]
log_alphas = [-k for k in range(3, 9)]
#dts = [delta_t(alpha) for alpha in alphas]
dts = [delta_t(10**k) for k in log_alphas]
#dts = [log10(delta_t(10**k)) for k in log_alphas]

#plot(alphas, dts)
plot(log_alphas, dts)

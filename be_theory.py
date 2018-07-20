import numpy as np
import matplotlib.pylab as plt
import pyhdust.phc as phc
from utils import find_nearest


# ==============================================================================
def obl2W(oblat):

    W = np.sqrt(2 * (oblat - 1))

    return W


# ==============================================================================
def t_tms_from_Xc(M, savefig=None, plot_fig=None, ttms_true=None, Xc=None):
    '''
    Calculates the t(tms) for a given Xc and mass
    Xc: float
    M: float
    '''
# ------------------------------------------------------------------------------
    # Parameters from the models
    mass = np.array([14.6, 12.5, 10.8, 9.6, 8.6, 7.7, 6.4, 5.5, 4.8,
                    4.2, 3.8, 3.4])

    nm = len(mass)
    str_mass = ['M14p60', 'M12p50', 'M10p80', 'M9p600', 'M8p600', 'M7p700',
                'M6p400', 'M5p500', 'M4p800', 'M4p200', 'M3p800', 'M3p400']
    st = ['B0.5', 'B1', 'B1.5', 'B2', 'B2.5', 'B3', 'B4', 'B5', 'B6', 'B7',
          'B8', 'B9']
    zsun = 'Z01400'
    str_vel = ['V60000', 'V70000', 'V80000', 'V90000', 'V95000']
    Hfracf = 0.  # end of main sequence

    # ****
    folder_data = 'tables/models/models_bes/'

    if plot_fig is True:
        plt.xlabel(r'$t/t_{MS}$')
        plt.ylabel(r'$X_c$')
        plt.ylim([0.0, 0.8])
        plt.xlim([0.0, 1.0])

# ------------------------------------------------------------------------------
    # Loop (reading the models)
    typ = (1, 3, 16, 21)  # Age, Lum versus Teff versus Hfrac
    arr_age = []
    arr_Hfr = []
    arr_t_tc = []
    cor = phc.gradColor(np.arange(len(st)), cmapn='inferno')
    iv = 2  # O que eh isto?
    arr_Xc = []
    for i in range(nm):
        file_data = folder_data + str_mass[i] + zsun + str_vel[iv] + '.dat'
        age, lum, Teff, Hfrac = np.loadtxt(file_data, usecols=typ,
                                           unpack=True, skiprows=2)
        arr_age.append(age)
        arr_Hfr.append(Hfrac)

        iMS = np.where(abs(Hfrac - Hfracf) == min(abs(Hfrac - Hfracf)))
        X_c = Hfrac[0:iMS[0][0]]
        arr_Xc.append(X_c)

        t_tc = age[0:iMS[0][0]] / max(age[0:iMS[0][0]])
        arr_t_tc.append(t_tc)
    if plot_fig is True:
        plt.plot(t_tc, X_c, color=cor[i], label=('%s' % st[i]))

# ------------------------------------------------------------------------------
# Interpolation
    k = find_nearest(mass, M)[1]

    if plot_fig is True:
        plt.plot(ttms_true, Xc, 'o')
        plt.autoscale()
        plt.minorticks_on()
        plt.legend(fontsize=10, ncol=2, fancybox=False, frameon=False)

# ------------------------------------------------------------------------------

    if savefig is True:
        pdfname = 'Xc_vs_Tsp.png'
        plt.savefig(pdfname)

    return k, arr_t_tc, arr_Xc

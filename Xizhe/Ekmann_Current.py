import numpy as np


c_o = 4100                          #specific heat capacity of seawater = 4100 Jkg^-1K^-1
omega = 2*np.pi/(24*3600)           #Earth's angular velocity
phi = 'LATITUDE'
f = 2*omega*np.sin(phi)            #Coriolis Parameter


def ekmann_current (tau_x, tau_y, dTm_dy, dTm_dx):
    Q_ek = c_o (tau_x*dTm_dy/f - tau_y*dTm_dx/f)
    return Q_ek


def ekmann_current_anom(tau_x_anom, tau_y_anom, dTm_bar_dy, dTm_bar_dx):
    Q_ek_anom = c_o(tau_x_anom*dTm_bar_dy/f - tau_y_anom*dTm_bar_dx/f)
    return Q_ek_anom


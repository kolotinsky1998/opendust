import numpy as np
from scipy.optimize import curve_fit


def E(z, E_0, z_critical, beta):
    return E_0 + z_critical*(-2*beta)*z + beta*z**2

def fit(z, E_z, z_min, z_max):
    l = z_max - z_min
    E_z_0 = E_z[0]
    z_dust_lowest = z[-1]
    E_z = E_z /  E_z_0
    z = z / l
    z_min = z_min / l
    z_max = z_max / l
    z_dust_lowest = z_dust_lowest / l
    popt, pcov = curve_fit(E, z, E_z, bounds=([-np.inf,max(z_dust_lowest,0.5*(z_min+z_max)),-np.inf], [np.inf, np.inf, 0]))
    E_0 = popt[0]*E_z_0
    alpha = -2*popt[2]*popt[1]*E_z_0/l
    beta = popt[2]*E_z_0/l**2
    return E_0, alpha, beta

if __name__ == "__main__":
    r_D_e = 0.0016622715189364113
    H = 5*r_D_e
    z_max = H/2
    z_min = -H/2 + r_D_e*0.5
    z = np.array([-1.94*r_D_e, -1.65*r_D_e, -1.37*r_D_e, -1.13*r_D_e, -0.89*r_D_e, -0.62*r_D_e, -0.33*r_D_e])
    z_dust_lowest = z[6]
    E_z = np.array([2251, 2431, 2386, 2528, 2597, 2539, 2607])
    E_0, alpha, beta = fit(z, E_z, z_min, z_max)

    for i in range(len(z)):
        print("{}\t{}".format(z[i], E_z[i]))

    print(E_0, alpha, beta)

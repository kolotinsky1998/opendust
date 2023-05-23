import sys
sys.path.insert(0,'/home/avtimofeev/opendust/chains')
sys.path.insert(0,'/home/avtimofeev/opendust/tools')
from fit import fit
from matthews_copy import DustParticle
from matthews_copy import PlasmaParametersInSIUnitsMatthews
from matthews_copy import SimulatioParametersInSIUnits
from matthews_copy import OutputParameters
from matthews_copy import OpenDust
import numpy as np
import time
from velocity import velocity

if __name__ == "__main__":
    iteration = 0
    ###############################################
    ### 1. Define plasma parameters in SI units ###
    ###############################################

    T_e = 30000  # electron temperature (K)
    T_n = 290  # ion temperature (K)
    n_inf = 1.0e14  # ion concentration (1/m^3)
    m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
    e = -1.60217662e-19
    eps_0 = 8.85418781762039e-12
    w_p_i = np.sqrt(n_inf*e**2/(eps_0*m_i))
    sigma_in = 5e-19 # ion-neutral collision cross-section (m^2)
    k_B = 1.380649e-23
    P = 18.131842 # Pressure (Pascal)
    v_B = np.sqrt(k_B * T_e / m_i)
    method = "quadratic"
    E_0, alpha, beta = 2600, 2396160.0, 0
    z_1 = 0
    z_2 = 0.005976353089443831*0.5
    externalFieldOrder = 2
    v_fl, w_c = velocity(E_0, P)
    M = v_fl/v_B
    Phi_parameters = {"E_0":[E_0,"V/m"],"alpha":[alpha,"V/m^2"],"beta":[beta,"V/m^3"],"z1":[z_1,"m"],"z7":[z_2,"m"]}

    plasmaParametersInSIUnits = PlasmaParametersInSIUnitsMatthews(
        T_n, T_e, n_inf, externalFieldOrder, Phi_parameters, M, w_c, m_i
    )
    plasmaParametersInSIUnits.printParameters()

    ###################################################
    ### 2. Define simulation parameters in SI units ###
    ###################################################

    R = 1.25 * plasmaParametersInSIUnits.r_D_e
    H = 5 * plasmaParametersInSIUnits.r_D_e
    N = int(2 ** 16)
    d_t_Equilibrate = min(0.1/w_p_i, 0.1/w_c)
    T = H/(M*v_B)
    nEquilibrate = int(4*T/d_t_Equilibrate)

    # object for equilibration
    simulationParametersInSIUnitsEquilibrate = SimulatioParametersInSIUnits(
        R, H, N, nEquilibrate, d_t_Equilibrate, plasmaParametersInSIUnits
    )

    simulationParametersInSIUnitsEquilibrate.printParameters()
    ###################################
    ### 3. Define output parameters ###
    ###################################

    directory = "/home/avtimofeev/opendust/examples/IonConcentration/data/"
    nOutputEquilibrate = 20
    nFileOutputEquilibrate = 20

    # object for equilibration
    csvOutputFileNameEquilibrate = directory + "csv/trajectoryEquilibrate_"
    xyzOutputFileNameEquilibrate = directory + "trajectoryEquilibrate.xyz"
    restartFileNameEquilibrate = directory + "RESTART_Equilibrate"
    outputParametersEquilibrate = OutputParameters(
        nOutputEquilibrate, nFileOutputEquilibrate, csvOutputFileNameEquilibrate, xyzOutputFileNameEquilibrate, restartFileNameEquilibrate
    )

    ################################
    ### 4. Define dust particles ###
    ################################

    dustParticles = []
    
    ###############################################################
    ### 5. e) Create OpenDust class object and start simulation ###
    ###       without dust particles for plasma equilibration   ###
    ###############################################################

    openDustEquilibrate = OpenDust(
        plasmaParametersInSIUnits,
        simulationParametersInSIUnitsEquilibrate,
        outputParametersEquilibrate,
        [],
    )

    startEquilibrate = time.time()
    openDustEquilibrate.simulate(deviceIndex = "0", cutOff = False, considerTrap = False)
    endEquilibrate = time.time()
    print("Equilibrate time: {}".format(endEquilibrate-startEquilibrate))
    

import sys
import os
import time
sys.path.insert(0,'/home/avtimofeev/opendust/chains')
sys.path.insert(0,'/home/avtimofeev/opendust/tools')
from fit import fit
from matthews import DustParticle
from matthews import PlasmaParametersInSIUnitsMatthews
from matthews import SimulatioParametersInSIUnits
from matthews import OutputParameters
from matthews import OpenDust
import numpy as np
import time
from velocity import velocity



if __name__ == "__main__":
   
    E_0, alpha, beta = 2600, 2396160.0, 0
    z_1 = 0
    z_2 = 0.005976353089443831*0.5

    ###############################################
    ### 1. Define plasma parameters in SI units ###
    ###############################################
    T_e = 30000  # electron temperature (K)
    T_n = 290  # ion temperature (K)
    n_inf = 1e14  # ion concentration (1/m^3)
    m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
    e = -1.60217662e-19
    eps_0 = 8.85418781762039e-12
    k_B = 1.380649e-23
    P = 18.131842 # Pressure (Pascal)
    w_p_i = np.sqrt(n_inf*e**2/(eps_0*m_i))
    tau_p_i = 2.0*np.pi/w_p_i
    v_B = np.sqrt(k_B * T_e / m_i)
    v_T = np.sqrt(k_B * T_n / m_i)

    method = "quadratic"
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
    d_t = min(0.1/w_p_i, 0.1/w_c)
    n = int(20)

    # object for modelling plasma around chain
    simulationParametersInSIUnits = SimulatioParametersInSIUnits(
        R, H, N, n, d_t, plasmaParametersInSIUnits
    )
    print("Main simulation parameters:")
    simulationParametersInSIUnits.printParameters()

    ###################################
    ### 3. Define output parameters ###
    ###################################
    directory = "/home/avtimofeev/opendust/examples/DustDynamics/TwoParticles/data/"
    nOutput = int(20)
    nFileOutput = int(20)
    # object for modelling plasma around chain
    csvOutputFileName = directory + "csv/trajectory{}_".format(0)
    xyzOutputFileName = directory + "trajectory{}.xyz".format(0)
    restartFileName = directory + "RESTART"
    trapFileName = directory + "trapFile.txt"
    outputParameters = OutputParameters(
        nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
    )
    os.system("touch {}trajectory.xyz".format(directory))
    for iteration in range(1, 55):
        print("Iteration: {}".format(iteration))

        ################################
        ### 4. Define dust particles ###
        ################################

        dustParticles = []

        ###################################################################
        ### 5. c) Create OpenDust class object and start simulation     ###
        ###       with dust particles to model plasma flow around chain ###
        ###################################################################

        openDust = OpenDust(
            plasmaParametersInSIUnits,
            simulationParametersInSIUnits,
            outputParameters,
            dustParticles,
        )

        start = time.time()
        openDust.simulate(deviceIndex = "0", cutOff = False, toRestartFileName = restartFileName, considerTrap = True, trapFileName = trapFileName)
        end = time.time()
        print("Main time: {}".format(end-start))
        os.system("cp {}trajectory.xyz {}trajectory_time.xyz".format(directory, directory))
        os.system("cat {}trajectory_time.xyz {} > {}trajectory.xyz".format(directory, xyzOutputFileName, directory))
        os.system("rm {}trajectory_time.xyz".format(directory))

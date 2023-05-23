import sys
sys.path.insert(0,'/home/avtimofeev/opendust/chains')
sys.path.insert(0,'/home/avtimofeev/opendust/tools')
from fit import fit
from matthews_copy_bare import DustParticle
from matthews_copy_bare import PlasmaParametersInSIUnitsMatthews
from matthews_copy_bare import SimulatioParametersInSIUnits
from matthews_copy_bare import OutputParameters
from matthews_copy_bare import OpenDust
import numpy as np
import time
from velocity import velocity



if __name__ == "__main__":
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
    r_p = 0.5*8.89e-6
    E_0, alpha, beta = 2600, 2396160.0, 0
    z_start_gradient = 0
    z_end_gradient = 0.5 * 0.005976353089443831
    method = "quadratic"
    externalFieldOrder = 2
    v_fl, w_c = velocity(E_0, P)
    M = v_fl/v_B
    Phi_parameters = {"E_0":[E_0,"V/m"],"alpha":[alpha,"V/m^2"],"beta":[beta,"V/m^3"],"z1":[z_start_gradient,"m"],"z7":[z_end_gradient,"m"]}

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
    d_t = min(0.1/w_p_i, 0.1*r_p/(M*v_B), 0.1/w_c)
    n = int(0.5 * tau_p_i/d_t/4.0)

    # object for modelling plasma around chain
    simulationParametersInSIUnits = SimulatioParametersInSIUnits(
        R, H, N, n, d_t, plasmaParametersInSIUnits
    )
    print("Main simulation parameters:")
    simulationParametersInSIUnits.printParameters()

    ###################################
    ### 3. Define output parameters ###
    ###################################
    directory = sys.argv[1] + "/"
    nOutput = int(10000)
    nFileOutput = int(10000)
    # object for modelling plasma around chain
    csvOutputFileName = directory + "csv/trajectory_"
    xyzOutputFileName = directory + "trajectory.xyz"
    restartFileName = directory + "RESTART"
    trapFileName = directory + "trapFile.txt"
    outputParameters = OutputParameters(
        nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, ""
    )

    

    ################################
    ### 4. Define dust particles ###
    ################################

    chargeCalculationMethod = "given"  # charge calculation method

    dustParticles = []
    _x_1, _y_1, _z_1, _r_1, _q_1 = 0, -H/8.0,  -H/4.0, 8.89e-6, -2e-15
    _x_2, _y_2, _z_2, _r_2, _q_2 = 0, -H/16.0,  -H/8.0, 8.89e-6, -2e-15
    _x_3, _y_3, _z_3, _r_3, _q_3 = 0, 0,  0, 8.89e-6, -2e-15
    _x_4, _y_4, _z_4, _r_4, _q_4 = 0, H/16.0,  H/8.0, 8.89e-6, -2e-15
    _x_5, _y_5, _z_5, _r_5, _q_5 = 0, H/8.0,  H/4.0, 8.89e-6, -2e-15
    
    dustParticles.append(DustParticle(_x_1, _y_1, _z_1, _r_1, chargeCalculationMethod, _q_1))
    dustParticles.append(DustParticle(_x_2, _y_2, _z_2, _r_2, chargeCalculationMethod, _q_2))
    dustParticles.append(DustParticle(_x_3, _y_3, _z_3, _r_3, chargeCalculationMethod, _q_3))
    dustParticles.append(DustParticle(_x_4, _y_4, _z_4, _r_4, chargeCalculationMethod, _q_4))
    dustParticles.append(DustParticle(_x_5, _y_5, _z_5, _r_5, chargeCalculationMethod, _q_5))

    ################################################################
    ### 5. Create OpenDust class object and start simulation     ###
    ###    with dust particles to model plasma flow around chain ###
    ################################################################

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

    ##############################################################
    ### 6. Calculate charge of microparticles and acting force ###
    ##############################################################

    t = openDust.t
    N_p = 5

    F_io = []
    F_z = []
    F_y = []

    t = openDust.t

    for p in range(N_p):
        F_io.append(openDust.dustParticles[p].forceIonsOrbit)
        F_z.append(np.zeros(n))
        F_y.append(np.zeros(n))

    for p in range(N_p):
        for i in range(n):
            F_z[p][i] = F_io[p][i][2]
            F_y[p][i] = F_io[p][i][1]

    
    f = open(directory + "force_y.txt", "w")
    for i in range(len(t)):
        f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(t[i],F_y[0][i],F_y[1][i],F_y[2][i],F_y[3][i],F_y[4][i]))
    f.close()

    f = open(directory + "force_z.txt", "w")
    for i in range(len(t)):
        f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(t[i],F_z[0][i],F_z[1][i],F_z[2][i],F_z[3][i],F_z[4][i]))
    f.close()
    print("Hello")
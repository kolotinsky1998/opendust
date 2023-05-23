import sys
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
    iteration = 0
    ###############################################
    ### 1. Define plasma parameters in SI units ###
    ###############################################

    T_e = 58022  # electron temperature (K)
    T_n = 290  # ion temperature (K)
    n_inf = 1e14  # ion concentration (1/m^3)
    m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
    e = -1.60217662e-19
    eps_0 = 8.85418781762039e-12
    w_p_i = np.sqrt(n_inf*e**2/(eps_0*m_i))
    k_B = 1.380649e-23
    P = 18.131842 # Pressure (Pascal)
    v_B = np.sqrt(k_B * T_e / m_i)
    r_p = 4.445e-6
    E = 4000
    v_fl, w_c = velocity(E, P)
    M = v_fl/v_B
    externalFieldOrder = 0
    Phi_parameters = {"E_0":[E,"V/m"]}

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
    d_t = min(0.1/w_p_i, 0.1*r_p/(M*v_B), 0.1/w_c)
    T = H/(M*v_B)
    nEquilibrate = int(2*T/d_t_Equilibrate)
    n = int(10*T/d_t)*4

    # object for equilibration
    simulationParametersInSIUnitsEquilibrate = SimulatioParametersInSIUnits(
        R, H, N, nEquilibrate, d_t_Equilibrate, plasmaParametersInSIUnits
    )

    # object for modelling plasma around chain
    simulationParametersInSIUnits = SimulatioParametersInSIUnits(
        R, H, N, n, d_t, plasmaParametersInSIUnits
    )
    simulationParametersInSIUnits.printParameters()


    ###################################
    ### 3. Define output parameters ###
    ###################################

    directory = sys.argv[1] + "/"
    nOutputEquilibrate = int(nEquilibrate/100)
    nFileOutputEquilibrate = int(nEquilibrate/100)
    nOutput = int(n/100)
    nFileOutput = int(n/100)
    # object for equilibration
    csvOutputFileNameEquilibrate = directory + "csv/trajectoryEquilibrate0_"
    xyzOutputFileNameEquilibrate = directory + "trajectoryEquilibrate0.xyz"
    restartFileNameEquilibrate = directory + "RESTART_Equilibrate"
    outputParametersEquilibrate = OutputParameters(
        nOutputEquilibrate, nFileOutputEquilibrate, csvOutputFileNameEquilibrate, xyzOutputFileNameEquilibrate, restartFileNameEquilibrate
    )
    # object for modelling plasma around chain
    csvOutputFileName = directory + "csv/trajectory0_"
    xyzOutputFileName = directory + "trajectory0.xyz"
    restartFileName = directory + "RESTART"
    outputParameters = OutputParameters(
        nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
    )

    ################################
    ### 4. Define dust particles ###
    ################################

    q = 0.0 * plasmaParametersInSIUnits.e  # charge of dust particles
    chargeCalculationMethod = "consistent"  # charge calculation method
    x_1, y_1, z_1, r_1, q_1 = 0, 0, -1.94 * plasmaParametersInSIUnits.r_D_e, r_p, q
    x_2, y_2, z_2, r_2, q_2 = 0, 0, -1.65 * plasmaParametersInSIUnits.r_D_e, r_p, q
    x_3, y_3, z_3, r_3, q_3 = 0, 0, -1.37 * plasmaParametersInSIUnits.r_D_e, r_p, q
    x_4, y_4, z_4, r_4, q_4 = 0, 0, -1.13 * plasmaParametersInSIUnits.r_D_e, r_p, q
    x_5, y_5, z_5, r_5, q_5 = 0, 0, -0.89 * plasmaParametersInSIUnits.r_D_e, r_p, q
    x_6, y_6, z_6, r_6, q_6 = 0, 0, -0.62 * plasmaParametersInSIUnits.r_D_e, r_p, q
    x_7, y_7, z_7, r_7, q_7 = 0, 0, -0.33 * plasmaParametersInSIUnits.r_D_e, r_p, q

    dustParticle1 = DustParticle(x_1, y_1, z_1, r_1, chargeCalculationMethod, q_1)
    dustParticle2 = DustParticle(x_2, y_2, z_2, r_2, chargeCalculationMethod, q_2)
    dustParticle3 = DustParticle(x_3, y_3, z_3, r_3, chargeCalculationMethod, q_3)
    dustParticle4 = DustParticle(x_4, y_4, z_4, r_4, chargeCalculationMethod, q_4)
    dustParticle5 = DustParticle(x_5, y_5, z_5, r_5, chargeCalculationMethod, q_5)
    dustParticle6 = DustParticle(x_6, y_6, z_6, r_6, chargeCalculationMethod, q_6)
    dustParticle7 = DustParticle(x_7, y_7, z_7, r_7, chargeCalculationMethod, q_7)

    dustParticle1 = DustParticle(x_1, y_1, z_1, r_1, chargeCalculationMethod, q_1)
    dustParticle2 = DustParticle(x_2, y_2, z_2, r_2, chargeCalculationMethod, q_2)

    dustParticles = [dustParticle1, dustParticle2, dustParticle3, dustParticle4, dustParticle5, dustParticle6, dustParticle7]
    
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
    openDustEquilibrate.simulate(deviceIndex = "0,1", cutOff = False, considerTrap = False)
    endEquilibrate = time.time()
    print("Equilibrate time: {}".format(endEquilibrate-startEquilibrate))
    
    ##################################################################
    ### 5. c) Create OpenDust class object and start simulation    ###
    ###       with dust particles to model plasma flow aroun chain ###
    ##################################################################

    openDust = OpenDust(
        plasmaParametersInSIUnits,
        simulationParametersInSIUnits,
        outputParameters,
        dustParticles,
    )

    start = time.time()
    openDust.simulate(deviceIndex = "0,1", cutOff = False, toRestartFileName = restartFileNameEquilibrate, considerTrap = True)
    end = time.time()
    print("Main time: {}".format(end-start))

    ##################
    ### 6. Analyze ###
    ##################

    N_p = 7 # number of dust particles
    g = 9.8 # free fall acceleration (m/s^2)
    rho = 1500
    m_p = 4.0/3.0*np.pi*r_p**3*rho # mass of dust particles (kg)

    n_averaging = int(n/2)

    F_io = []
    F_ic = []
    F_dd = []
    F_E = []
    E_z = []
    F_z = []

    meanF_z = np.zeros(N_p)
    meanE_z = np.zeros(N_p)
    errorF_z = np.zeros(N_p)
    errorE_z = np.zeros(N_p)
    meanQ = np.zeros(N_p)
    errorQ = np.zeros(N_p)
    iterator = 0
    

    t = openDust.t
    q = []

    for p in range(N_p):
        F_io.append(openDust.dustParticles[p].forceIonsOrbit)
        F_ic.append(openDust.dustParticles[p].forceIonsCollect)
        F_dd.append(openDust.dustParticles[p].forceDust)
        F_E.append(openDust.dustParticles[p].forceExternalField)
        E_z.append(np.zeros(n))
        F_z.append(np.zeros(n))
        q.append(openDust.dustParticles[p].q)

    for p in range(N_p):
        for i in range(n):
            E_z[p][i] = -(F_io[p][i][2] + F_ic[p][i][2] + F_dd[p][i][2] + m_p * g) / q[p][i]
            F_z[p][i] = (F_E[p][i][2] + F_io[p][i][2] + F_ic[p][i][2] + F_dd[p][i][2] + m_p * g) / (m_p * g)

    for p in range(N_p):
        for i in range(n):
            if i > n_averaging:
                iterator += 1
                meanF_z[p] += F_z[p][i]
                meanE_z[p] += E_z[p][i]
                meanQ[p] += q[p][i]
        meanF_z[p] /= float(iterator)
        meanE_z[p] /= float(iterator)
        meanQ[p] /= float(iterator)
        iterator = 0
    
    for p in range(N_p):
        for i in range(n):
            if i > n_averaging:
                iterator += 1
                errorF_z[p] += (F_z[p][i] - meanF_z[p])**2
                errorE_z[p] += (E_z[p][i] - meanE_z[p])**2
                errorQ[p] += (q[p][i] - meanQ[p])**2
        errorF_z[p] = np.sqrt(errorF_z[p]) / np.sqrt(float(iterator))
        errorE_z[p] = np.sqrt(errorE_z[p]) / np.sqrt(float(iterator))
        errorQ[p] = np.sqrt(errorQ[p]) / np.sqrt(float(iterator))
        iterator = 0
    
    f = open(directory + "charge{}.txt".format(iteration), "w")
    for p in range(N_p):
        for i in range(n):
            f.write("{}\t{}\n".format(t[i],q[p][i]))
        f.write("\n\n")
    f.close()

    f = open(directory + "force{}.txt".format(iteration), "w")
    for p in range(N_p):
        for i in range(n):
            f.write("{}\t{}\t{}\t{}\t{}\n".format(t[i], F_io[p][i][2], F_ic[p][i][2], F_dd[p][i][2], F_E[p][i][2]))
        f.write("\n\n")
    f.close()

    z = np.array([z_1, z_2, z_3, z_4, z_5, z_6, z_7])

    f = open(directory + "data{}.txt".format(iteration), "w")
    f.write("# z\t mean\t std\n")
    f.write("# charge\n")
    for p in range(N_p):
        f.write("{}\t{}\t{}\n".format(z[p],meanQ[p],errorQ[p]))
    f.write("\n\n")
    f.write("# E_z\n")
    for p in range(N_p):
        f.write("{}\t{}\t{}\n".format(z[p],meanE_z[p],errorE_z[p]))
    f.write("\n\n")
    f.write("# F_T / F_g\n")
    for p in range(N_p):
        f.write("{}\t{}\t{}\n".format(z[p],meanF_z[p],errorF_z[p]))
    f.close()
    

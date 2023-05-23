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
    g = 9.8 # free fall acceleration (m/s^2)
    rho = 1500 
    n_dust = 20000
    T_p = 300
    d_t_p = 5e-5 # integration step for dust particles dynamics, s
    N_p = 2 # number of dust particles
    x, y, z = np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust))
    v_x, v_y, v_z = np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust))
    f_x, f_y, f_z = np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust))
    t = np.zeros(n_dust)
    r = np.zeros(N_p)
    q = np.zeros((N_p, n_dust))
    z[0][0] = -0.5*0.25e-3
    z[1][0] = 0.5*0.25e-3
    q[0][0] = 0
    q[1][0] = 0
    r[0] = 0.5*8.89e-6
    r[1] = 0.5*8.89e-6
    f_x[0][0], f_y[0][0], f_z[0][0] = 0, 0, 0
    v_x[0][0], v_y[0][0], v_z[0][0] = 0, 0, 0
    f_x[1][0], f_y[1][0], f_z[1][0] = 0, 0, 0
    v_x[1][0], v_y[1][0], v_z[1][0] = 0, 0, 0
    t[0] = 0

    E_0, alpha, beta = 2600, 2396160.0, 0
    z_start_gradient = 0
    z_end_gradient = 0.5 * 0.005976353089443831

    dustInfoFileName = []
    for p in range(N_p):
        dustInfoFileName.append(sys.argv[1] + "/dust_info_{}".format(p))

    for p in range(N_p):
        file = open(dustInfoFileName[p], "w")
        file.close()

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
    r_p = r[0]

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
    n = int(0.5 * tau_p_i/d_t)

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
        nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
    )

    for iteration in range(1, n_dust):
        print("Iteration: {}".format(iteration))

        ################################
        ### 4. Define dust particles ###
        ################################

        chargeCalculationMethod = "consistent"  # charge calculation method

        dustParticles = []
        for p in range(N_p):
            _x, _y, _z, _r, _q = x[p][iteration-1], y[p][iteration-1], z[p][iteration-1], r[p], q[p][iteration-1]
            dustParticles.append(DustParticle(_x, _y, _z, _r, chargeCalculationMethod, _q))

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
        openDust.simulate(deviceIndex = "0,1", cutOff = False, toRestartFileName = restartFileName, considerTrap = True, trapFileName = trapFileName)
        end = time.time()
        print("Main time: {}".format(end-start))

        ##############################################################
        ### 6. Calculate charge of microparticles and acting force ###
        ##############################################################

        for p in range(N_p):
            _q = 0
            iterator = 0
            for i in range(n):
                iterator += 1
                _q += openDust.dustParticles[p].q[i]
            _q /= float(iterator)
            q[p][iteration] = 0.75*q[p][iteration-1] + 0.25*_q

        for p in range(N_p):
            _f_x = 0
            iterator = 0
            for i in range(n):
                iterator += 1
                _f_x += openDust.dustParticles[p].forceIonsOrbit[i][0]
                _f_x += openDust.dustParticles[p].forceIonsCollect[i][0]
                _f_x += openDust.dustParticles[p].forceDust[i][0]
                _f_x += openDust.dustParticles[p].forceExternalField[i][0]
            _f_x /= float(iterator)
            f_x[p][iteration] = 0.75*f_x[p][iteration-1] + 0.25*_f_x

        for p in range(N_p):
            _f_y = 0
            iterator = 0
            for i in range(n):
                iterator += 1
                _f_y += openDust.dustParticles[p].forceIonsOrbit[i][1]
                _f_y += openDust.dustParticles[p].forceIonsCollect[i][1]
                _f_y += openDust.dustParticles[p].forceDust[i][1]
                _f_y += openDust.dustParticles[p].forceExternalField[i][1]
            _f_y /= float(iterator)
            f_y[p][iteration] = 0.75*f_y[p][iteration-1] + 0.25*_f_y

        for p in range(N_p):
            _f_z = 0
            iterator = 0
            for i in range(n):
                iterator += 1
                _f_z += openDust.dustParticles[p].forceIonsOrbit[i][2]
                _f_z += openDust.dustParticles[p].forceIonsCollect[i][2]
                _f_z += openDust.dustParticles[p].forceDust[i][2]
                _f_z += openDust.dustParticles[p].forceExternalField[i][2]
            _f_z /= float(iterator)
            f_z[p][iteration] = 0.75*f_z[p][iteration-1] + 0.25*_f_z

        #################################################
        ### 7. Calculate new dust particles positions ###
        #################################################

        for p in range(N_p):
            m_p = 4.0/3.0*np.pi*r[p]**3*rho * 1.05 # mass of dust particles (kg)
            gamma_p = 8.0/3.0*np.sqrt(2.0*np.pi)*r[p]**2*P/v_T/m_p
            s_p = np.sqrt(2 * k_B * T_p * m_p * gamma_p / d_t_p)
            f_therm_p_x = np.random.normal(0, s_p, 1)[0] - m_p*gamma_p*v_x[p][iteration-1]
            f_therm_p_y = np.random.normal(0, s_p, 1)[0] - m_p*gamma_p*v_y[p][iteration-1]
            f_therm_p_z = np.random.normal(0, s_p, 1)[0] - m_p*gamma_p*v_z[p][iteration-1]
            _a_x = (f_therm_p_x + f_x[p][iteration]) / m_p
            _a_y = (f_therm_p_y + f_y[p][iteration]) / m_p
            _a_z = (f_therm_p_z + f_z[p][iteration]) / m_p + g
            x[p][iteration] = x[p][iteration-1] + v_x[p][iteration-1] * d_t_p + 0.5*_a_x * d_t_p **2
            y[p][iteration] = y[p][iteration-1] + v_y[p][iteration-1] * d_t_p + 0.5*_a_y * d_t_p **2
            z[p][iteration] = z[p][iteration-1] + v_z[p][iteration-1] * d_t_p + 0.5*_a_z * d_t_p **2

            v_x[p][iteration] =  v_x[p][iteration-1] + _a_x * d_t_p
            v_y[p][iteration] =  v_y[p][iteration-1] + _a_y * d_t_p
            v_z[p][iteration] =  v_z[p][iteration-1] + _a_z * d_t_p

            t[iteration] = t[iteration-1] + d_t_p
    
        for p in range(N_p):
            file = open(dustInfoFileName[p], "a")
            file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(t[iteration], x[p][iteration], y[p][iteration], z[p][iteration], v_x[p][iteration], v_y[p][iteration], v_z[p][iteration], f_x[p][iteration], f_y[p][iteration], f_z[p][iteration], q[p][iteration]))
            file.close()
            
        


        
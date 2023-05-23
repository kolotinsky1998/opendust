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

    ###############################################
    ### 1. Define plasma parameters in SI units ###
    ###############################################

    T_e = 58022  # electron temperature (K)
    T_n = 290  # ion temperature (K)
    n_inf = 1e14  # ion concentration (1/m^3)
    m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
    e = -1.60217662e-19 # electron charge
    eps_0 = 8.85418781762039e-12 # vacuum dielectric permitivity
    w_p_i = np.sqrt(n_inf*e**2/(eps_0*m_i)) # ion plasma frequency (1/s)
    tau_p_i = 2.0*np.pi/w_p_i # ion plasma period (s)
    k_B = 1.380649e-23 # Boltzmann constant (J/K)
    P = 18.131842 # Pressure (Pascal)
    v_B = np.sqrt(k_B * T_e / m_i) # Bohm's velocity (m/s)
    r_D_e = np.sqrt(eps_0*k_B*T_e/n_inf/e**2) # Electron Debey radius (m)
    H = 5*r_D_e # Height of the computational domain (m)
    method = "quadratic"
    E_0 = 2614.9630526115625
    alpha = 35939.98381792599
    beta = -43242013.60423763
    z_min = -H/2.0 + 0.5*r_D_e
    z_max = H/2.0
    externalFieldOrder = 2
    v_fl, w_c = velocity(E_0, P)
    M = v_fl/v_B # Mach number of the ion flow
    Phi_parameters = {"E_0":[E_0,"V/m"],"alpha":[alpha,"V/m^2"],"beta":[beta,"V/m^3"],"z1":[z_min,"m"],"z7":[z_max,"m"]}

    plasmaParametersInSIUnits = PlasmaParametersInSIUnitsMatthews(
        T_n, T_e, n_inf, externalFieldOrder, Phi_parameters, M, w_c, m_i
    )
    plasmaParametersInSIUnits.printParameters()

    ###################################################
    ### 2. Define simulation parameters in SI units ###
    ###################################################

    r_p = 4.445e-6 # Dust particle radius
    R = 1.25 * plasmaParametersInSIUnits.r_D_e # Radius of the computational domain
    H = 5 * plasmaParametersInSIUnits.r_D_e # Height of the computational domain
    N = int(2 ** 16) # Number of superions
    d_t = min(0.1/w_p_i, 0.1*r_p/(M*v_B), 0.1/w_c) # Ion integration time-step
    n = 3340 # Number of ion integration time-step

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
    
    csvOutputFileName = directory + "csv/trajectory_"
    xyzOutputFileName = directory + "trajectory.xyz"
    restartFileName = directory + "RESTART"
    trapFileName = directory + "trapFile.txt"
    outputParameters = OutputParameters(
        nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
    )

    #########################################################
    ### 4.0 Define some dust particles related parameters ###
    #########################################################

    g = 9.8 # free fall acceleration (m/s^2)
    rho = 1500 # mass density of dust particle 
    n_dust = 10000 # number of dust particles time steps
    T_p = 2000 # Kinetic temperature of dust particles motion
    k_x = 2e-9 # x-trap (kg/s^2)
    k_y = 2e-9 # y-trap (kg/s^2)
    d_t_p = 5e-5 # integration step for dust particles dynamics, s
    N_p = 7 # number of dust particles
    coefficient = 0.1 # Lag coefficient for charges and forces averaging

    x_1, y_1, z_1, r_1, q_1 = 0, 0, -1.94 * plasmaParametersInSIUnits.r_D_e, r_p, -2.764690497346931e-15
    x_2, y_2, z_2, r_2, q_2 = 0, 0, -1.65 * plasmaParametersInSIUnits.r_D_e, r_p, -2.4840083760225103e-15
    x_3, y_3, z_3, r_3, q_3 = 0, 0, -1.37 * plasmaParametersInSIUnits.r_D_e, r_p, -2.4938555990669545e-15
    x_4, y_4, z_4, r_4, q_4 = 0, 0, -1.13 * plasmaParametersInSIUnits.r_D_e, r_p, -2.4563210594518954e-15
    x_5, y_5, z_5, r_5, q_5 = 0, 0, -0.89 * plasmaParametersInSIUnits.r_D_e, r_p, -2.4364653691297227e-15
    x_6, y_6, z_6, r_6, q_6 = 0, 0, -0.62 * plasmaParametersInSIUnits.r_D_e, r_p, -2.48668278563173e-15
    x_7, y_7, z_7, r_7, q_7 = 0, 0, -0.33 * plasmaParametersInSIUnits.r_D_e, r_p, -2.528894600889228e-15

    x, y, z = np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust))
    v_x, v_y, v_z = np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust))
    f_x, f_y, f_z = np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust)), np.zeros((N_p, n_dust))
    t = np.zeros(n_dust)
    r = np.zeros(N_p)
    q = np.zeros((N_p, n_dust))

    x[0][0], x[1][0], x[2][0], x[3][0], x[4][0], x[5][0], x[6][0] = x_1, x_2, x_3, x_4, x_5, x_6, x_7
    y[0][0], y[1][0], y[2][0], y[3][0], y[4][0], y[5][0], y[6][0] = y_1, y_2, y_3, y_4, y_5, y_6, y_7
    z[0][0], z[1][0], z[2][0], z[3][0], z[4][0], z[5][0], z[6][0] = z_1, z_2, z_3, z_4, z_5, z_6, z_7
    q[0][0], q[1][0], q[2][0], q[3][0], q[4][0], q[5][0], q[6][0] = q_1, q_2, q_3, q_4, q_5, q_6, q_7
    r[0], r[1], r[2], r[3], r[4], r[5], r[6] = r_1, r_2, r_3, r_4, r_5, r_6, r_7

    m = 4.0/3.0*np.pi*r**3*rho # mass of dust particles (kg)

    dustInfoFileName = []
    for p in range(N_p):
        dustInfoFileName.append(directory + "dust_info_{}".format(p))

    for p in range(N_p):
        file = open(dustInfoFileName[p], "w")
        file.close()

    q_last_iteration = []
    for p in range(N_p):
        q_last_iteration.append(q[p][0])
        
    for iteration in range(1, n_dust):
        print("Iteration: {}".format(iteration))
        
        ################################
        ### 4. Define dust particles ###
        ################################

        chargeCalculationMethod = "consistent"  # charge calculation method

        dustParticles = []
        for p in range(N_p):
            _x, _y, _z, _r, _q = x[p][iteration-1], y[p][iteration-1], z[p][iteration-1], r[p], q_last_iteration[p]
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
        openDust.simulate(deviceIndex = "0,1,2,3", cutOff = False, toRestartFileName = restartFileName, considerTrap = True, trapFileName = trapFileName)
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
            q[p][iteration] = (1.0-coefficient)*q[p][iteration-1] + coefficient*_q
            q_last_iteration[p] = openDust.dustParticles[p].q[n-1]

        for p in range(N_p):
            _f_x = 0
            iterator = 0
            for i in range(n):
                iterator += 1
                _f_x += openDust.dustParticles[p].forceIonsOrbit[i][0]
                _f_x += openDust.dustParticles[p].forceIonsCollect[i][0]
                _f_x += openDust.dustParticles[p].forceDust[i][0]
                _f_x += -k_x*x[p][iteration-1]
            _f_x /= float(iterator)
            f_x[p][iteration] = (1.0-coefficient)*f_x[p][iteration-1] + coefficient*_f_x

        for p in range(N_p):
            _f_y = 0
            iterator = 0
            for i in range(n):
                iterator += 1
                _f_y += openDust.dustParticles[p].forceIonsOrbit[i][1]
                _f_y += openDust.dustParticles[p].forceIonsCollect[i][1]
                _f_y += openDust.dustParticles[p].forceDust[i][1]
                _f_y += -k_y*y[p][iteration-1]
            _f_y /= float(iterator)
            f_y[p][iteration] = (1.0-coefficient)*f_y[p][iteration-1] + coefficient*_f_y

        for p in range(N_p):
            _f_z = 0
            iterator = 0
            for i in range(n):
                iterator += 1
                _f_z += openDust.dustParticles[p].forceIonsOrbit[i][2]
                _f_z += openDust.dustParticles[p].forceIonsCollect[i][2]
                _f_z += openDust.dustParticles[p].forceDust[i][2]
                _f_z += q[p][iteration-1] * (E_0 + alpha*z[p][iteration-1] + beta*z[p][iteration-1]**2)
            _f_z /= float(iterator)
            f_z[p][iteration] = (1.0-coefficient)*f_z[p][iteration-1] + coefficient*_f_z
        
        #################################################
        ### 7. Calculate new dust particles positions ###
        #################################################

        for p in range(N_p):
            m_p = m[p]
            v_T = np.sqrt(k_B * T_p / m_i)
            gamma_p = 10*8.0/3.0*np.sqrt(2.0*np.pi)*r[p]**2*P/v_T/m_p
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
            if z[p][iteration] > 0:
                z[p][iteration] = - z[p][iteration]
                v_z[p][iteration] = - v_z[p][iteration]
            v_x[p][iteration] =  v_x[p][iteration-1] + _a_x * d_t_p
            v_y[p][iteration] =  v_y[p][iteration-1] + _a_y * d_t_p
            v_z[p][iteration] =  v_z[p][iteration-1] + _a_z * d_t_p

            t[iteration] = t[iteration-1] + d_t_p
    
        for p in range(N_p):
            file = open(dustInfoFileName[p], "a")
            file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(t[iteration], x[p][iteration], y[p][iteration], z[p][iteration], v_x[p][iteration], v_y[p][iteration], v_z[p][iteration], f_x[p][iteration], f_y[p][iteration], f_z[p][iteration] + g * m[p], q[p][iteration]))
            file.close()
            
        


        
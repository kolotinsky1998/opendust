import sys
sys.path.insert(0,'/home/avtimofeev/opendust/chains')
sys.path.insert(0,'/home/avtimofeev/opendust/tools')
from fit import fit
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

    ###################################################
    ### 2. Define simulation parameters in SI units ###
    ###################################################

    ###################################
    ### 3. Define output parameters ###
    ###################################

    for iteration in range(1, n_dust):
        print("Iteration: {}".format(iteration))

        ################################
        ### 4. Define dust particles ###
        ################################

        ################################################################
        ### 5. Create OpenDust class object and start simulation     ###
        ###    with dust particles to model plasma flow around chain ###
        ################################################################

        ##############################################################
        ### 6. Calculate charge of microparticles and acting force ###
        ##############################################################

        def F_E(z, q, E_0, alpha):
            if z < 0:
                return E_0*q
            else:
                return (E_0 + alpha*z)*q
        def F_q(z1, z2, q1, q2):
            # 2 acts on 1
            k = 9*1.0e9
            return k*q1*q2*(z1-z2)/(abs(z1-z2))**3.0
        
        for p in range(N_p):
            f_x[p][iteration] = 0
            f_y[p][iteration] = 0
            q[p][iteration] = -2.15e-15
        f_z[0][iteration] = F_E(z[0][iteration-1], q[0][iteration], E_0, alpha) +  F_q(z[0][iteration-1], z[1][iteration-1], q[0][iteration], q[1][iteration])
        f_z[1][iteration] = F_E(z[1][iteration-1], q[1][iteration], E_0, alpha) +  F_q(z[1][iteration-1], z[0][iteration-1], q[1][iteration], q[0][iteration])

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
        file = open(dustInfoFileName[p], "w")
        for i in range(n_dust):
            file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(t[i], x[p][i], y[p][i], z[p][i], v_x[p][i], v_y[p][i], v_z[p][i], f_x[p][i], f_y[p][i], f_z[p][i], q[p][i]))
        file.close()
            
        


        
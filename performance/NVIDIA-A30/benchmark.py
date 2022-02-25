from opendust.opendust import DustParticle
from opendust.opendust import PlasmaParametersInSIUnitsMaxwell
from opendust.opendust import SimulatioParametersInSIUnits
from opendust.opendust import OutputParameters
from opendust.opendust import OpenDust
import sys
import time


if __name__ == "__main__":

    ########################################
    ### 1. Plasma parameters in SI units ###
    ########################################

    T_e = 29011  # electron temperature (K)
    T_i = 290.11  # ion temperature (K)
    n_inf = 1e14  # ion concentration (1/m^3)
    m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
    M = 1  # Mach number of the ion flow

    distributionType = "Maxwellian"
    plasmaParametersInSIUnits = PlasmaParametersInSIUnitsMaxwell(
        T_i, T_e, n_inf, M, m_i
    )

    ############################################
    ### 2. Simulation parameters in SI units ###
    ############################################

    R = 3 * plasmaParametersInSIUnits.r_D_e
    H = 6 * plasmaParametersInSIUnits.r_D_e
    N = int(2 ** float((sys.argv[1])))
    n = 3000
    d_t = 3.5148240854e-09
    simulationParametersInSIUnits = SimulatioParametersInSIUnits(
        R, H, N, n, d_t, plasmaParametersInSIUnits
    )

    ###################################
    ### 3. Define output parameters ###
    ###################################

    nOutput = 2000
    nFileOutput = 2000

    outputParameters = OutputParameters(
        nOutput, nFileOutput
    )

    ################################
    ### 4. Define dust particles ###
    ################################

    r = 58.8e-7  # radius of dust particles
    q = 392500.0 * plasmaParametersInSIUnits.e  # charge of dust particles
    chargeCalculationMethod = "given"  # charge calculation method

    x_1, y_1, z_1, r_1, q_1 = (
        0 * plasmaParametersInSIUnits.r_D_e,
        0,
        -1 * plasmaParametersInSIUnits.r_D_e,
        r,
        q,
    )

    dustParticle1 = DustParticle(x_1, y_1, z_1, r_1, chargeCalculationMethod, q_1)

    dustParticles = [dustParticle1]

    ############################################################
    ### 5. Create OpenDust class object and start simulation ###
    ############################################################

    openDust = OpenDust(
        plasmaParametersInSIUnits,
        simulationParametersInSIUnits,
        outputParameters,
        dustParticles,
        distributionType,
    )
    start = time.time()
    openDust.simulate()
    end = time.time()
    file = open("time.txt",'a')
    file.write("{}\t{}\n".format(sys.argv[1], end-start))
    file.close()
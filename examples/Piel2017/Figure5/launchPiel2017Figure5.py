from opendust import opendust
from opendust import DustParticle
from opendust import PlasmaParametersInSIUnitsMaxwell
from opendust import PlasmaParametersInSIUnitsFieldDriven
from opendust import SimulatioParametersInSIUnits
from opendust import OutputParameters
from opendust import OpenDust

import math
import numpy as np

if __name__ == "__main__":

    M_array = np.asarray(
        [
            0.19679,
            0.30120,
            0.40161,
            0.49799,
            0.60241,
            0.69880,
            0.80321,
            0.89960,
            0.99598,
            1.19679,
            1.49799,
            2.00000,
        ]
    )
    for p in range(12):
        ########################################
        ### 1. Plasma parameters in SI units ###
        ########################################

        T_e = 29011  # electron temperature (K)
        T_n = 290.11  # neutral gas temperature (K)
        n_inf = 1e14  # ion concentration (1/m^3)
        m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
        M = M_array[p] #Mach number of the ion flow

        distributionType = "Maxwellian"
        plasmaParametersInSIUnits = PlasmaParametersInSIUnitsMaxwell(
            T_n, T_e, n_inf, M, m_i
        )
        plasmaParametersInSIUnits.printParameters()

        ############################################
        ### 2. Simulation parameters in SI units ###
        ############################################

        R = 3 * plasmaParametersInSIUnits.r_D_e
        H = 6 * plasmaParametersInSIUnits.r_D_e
        N = int(2 ** 16)
        n = 5000
        d_t = 3.5148240854e-09
        simulationParametersInSIUnits = SimulatioParametersInSIUnits(
            R, H, N, n, d_t, plasmaParametersInSIUnits
        )
        simulationParametersInSIUnits.printParameters()

        ############################
        ### 3. Output parameters ###
        ############################

        directory = "/home/avtimofeev/kolotinskii/opendust/data/Piel2017/Figure5/point{}/".format(p+1)
        nOutput = 1000
        nFileOutput = 1000
        csvOutputFileName = directory + "csv/trajectory"
        xyzOutputFileName = directory + "trajectory.xyz"
        restartFileName = directory + "RESTART"
        outputParameters = OutputParameters(
            nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
        )

        #########################
        ### 4. Dust particles ###
        #########################

        r = 58.8e-6  # radius of dust particles
        q = 392500.0 * plasmaParametersInSIUnits.e  # charge of dust particles
        chargeCalculationMethod = "oml"  # charge calculation method

        x_1, y_1, z_1, r_1, q_1 = 0, 0, 0 * plasmaParametersInSIUnits.r_D_e, r, q

        dustParticle1 = DustParticle(x_1, y_1, z_1, r_1, chargeCalculationMethod, q_1)

        dustParticles = [dustParticle1]

        ##################################
        ### 5. Create open dust object ###
        ##################################

        openDust = OpenDust(
            plasmaParametersInSIUnits,
            simulationParametersInSIUnits,
            outputParameters,
            dustParticles,
            distributionType,
        )
        #########################
        ### 6. Start dynamics ###
        #########################

        platformName = "CUDA"
        toRestartFileName = ""
        openDust.simulate(platformName, toRestartFileName)

        ##################
        ### 7. Analyze ###
        ##################

        forceIonsOrbitZ = openDust.dustParticles[0].forceIonsOrbit
        forceIonsCollectZ = openDust.dustParticles[0].forceIonsCollect
        q = openDust.dustParticles[0].q
        t = openDust.t

        f = open(directory+"force.txt","w")

        for i in range(n):
            f.write(
                "{}\t{}\t{}\n".format(
                    t[i],
                    forceIonsOrbitZ[i][2],
                    forceIonsCollectZ[i][2],
                )
            )
        f.close()

        meanForceIonsOrbitZ = 0
        meanForceIonsCollectZ = 0

        iterator = 0

        for i in range(n):
            if i > 3000:
                iterator += 1
                meanForceIonsOrbitZ += forceIonsOrbitZ[i][2]
                meanForceIonsCollectZ += forceIonsCollectZ[i][2]

        meanForceIonsOrbitZ /= iterator
        meanForceIonsCollectZ /= iterator

        print(
            "{}\t{}\t{}".format(
                M,
                meanForceIonsOrbitZ,
                meanForceIonsCollectZ,
            )
        )

        f = open(directory+"charge.txt","w")

        for i in range(n):
            f.write(
                "{}\t{}\n".format(
                    t[i],
                    q[i]
                )
            )
        f.close()

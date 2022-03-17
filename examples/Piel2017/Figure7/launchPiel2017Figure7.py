from opendust.opendust import DustParticle
from opendust.opendust import PlasmaParametersInSIUnitsMaxwell
from opendust.opendust import SimulatioParametersInSIUnits
from opendust.opendust import OutputParameters
from opendust.opendust import OpenDust

import numpy as np

if __name__ == "__main__":

    d_x_array = np.asarray(
        [
            0.00176,
            0.10369,
            0.20211,
            0.30053,
            0.40246,
            0.50088,
            0.60105,
            0.70123,
            0.80141,
            0.89982,
            1.00000,
        ]
    )
    for p in range(11):
        ###############################################
        ### 1. Define plasma parameters in SI units ###
        ###############################################

        T_e = 29011  # electron temperature (K)
        T_n = 290.11  # neutral gas temperature (K)
        n_inf = 1e14  # ion concentration (1/m^3)
        m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
        M = 1  # Mach number of the ion flow

        distributionType = "Maxwellian"
        plasmaParametersInSIUnits = PlasmaParametersInSIUnitsMaxwell(
            T_n, T_e, n_inf, M, m_i
        )
        plasmaParametersInSIUnits.printParameters()

        ###################################################
        ### 2. Define simulation parameters in SI units ###
        ###################################################

        R = 3 * plasmaParametersInSIUnits.r_D_e
        H = 6 * plasmaParametersInSIUnits.r_D_e
        N = int(2 ** 17)
        n = 5000
        d_t = 0.5148240854e-09
        simulationParametersInSIUnits = SimulatioParametersInSIUnits(
            R, H, N, n, d_t, plasmaParametersInSIUnits
        )
        simulationParametersInSIUnits.printParameters()

        ###################################
        ### 3. Define output parameters ###
        ###################################

        directory = "/home/avtimofeev/opendust/data/Piel2017/Figure7/point{}/".format(p + 1)
        nOutput = 1000
        nFileOutput = 1000
        csvOutputFileName = directory + "csv/trajectory"
        xyzOutputFileName = directory + "trajectory.xyz"
        restartFileName = directory + "RESTART"
        outputParameters = OutputParameters(
            nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
        )

        ################################
        ### 4. Define dust particles ###
        ################################
        
        r = 25.0e-6  # radius of dust particles
        q = 170000.0 * plasmaParametersInSIUnits.e  # charge of dust particles
        chargeCalculationMethod = "given"  # charge calculation method

        x_1, y_1, z_1, r_1, q_1 = 0, 0, -0.5 * plasmaParametersInSIUnits.r_D_e, r, q
        x_2, y_2, z_2, r_2, q_2 = (
            d_x_array[p] * plasmaParametersInSIUnits.r_D_e,
            0,
            0.5 * plasmaParametersInSIUnits.r_D_e,
            r,
            q,
        )

        dustParticle1 = DustParticle(x_1, y_1, z_1, r_1, chargeCalculationMethod, q_1)
        dustParticle2 = DustParticle(x_2, y_2, z_2, r_2, chargeCalculationMethod, q_2)

        dustParticles = [dustParticle1, dustParticle2]

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

        openDust.simulate(deviceIndex = "0,1,2,3,4,5,6,7", cutOff = False)

        ##################
        ### 7. Analyze ###
        ##################

        forceIonsOrbit1 = openDust.dustParticles[0].forceIonsOrbit
        forceDust1 = openDust.dustParticles[0].forceDust
        forceIonsOrbit2 = openDust.dustParticles[1].forceIonsOrbit
        forceDust2 = openDust.dustParticles[1].forceDust
        t = openDust.t

        f = open(directory + "force.txt", "w")

        for i in range(n):
            f.write(
                "{}\t{}\t{}\t{}\t{}\n".format(
                    t[i],
                    forceIonsOrbit1[i][0] + forceDust1[i][0],
                    forceIonsOrbit1[i][2] + forceDust1[i][2],
                    forceIonsOrbit2[i][0] + forceDust2[i][0],
                    forceIonsOrbit2[i][2] + forceDust2[i][2],
                )
            )
        f.close()

        meanForce1X = 0
        meanForce1Z = 0
        meanForce2X = 0
        meanForce2Z = 0

        iterator = 0

        for i in range(n):
            if i > 2500:
                iterator += 1
                meanForce1X += forceIonsOrbit1[i][0] + forceDust1[i][0]
                meanForce1Z += forceIonsOrbit1[i][2] + forceDust1[i][2]
                meanForce2X += forceIonsOrbit2[i][0] + forceDust2[i][0]
                meanForce2Z += forceIonsOrbit2[i][2] + forceDust2[i][2]

        meanForce1X /= iterator
        meanForce1Z /= iterator
        meanForce2X /= iterator
        meanForce2Z /= iterator

        print(
            "{}\t{}\t{}\t{}\t{}".format(
                d_x_array[p], meanForce1X, meanForce1Z, meanForce2X, meanForce2Z,
            )
        )


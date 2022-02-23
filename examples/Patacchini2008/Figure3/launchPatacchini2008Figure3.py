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
    w_c_array = np.asarray(
        [
            1.57319303e04,
            1.57319303e05,
            4.87689840e05,
            1.66758461e06,
            4.93982612e06,
            1.64084033e07,
            5.00275384e07,
            1.65562835e08,
            4.89121446e08,
            1.65767350e09,
            3.22479401e09,
            8.05204244e09,
            1.61113688e10,
        ]
    )
    E_array = np.asarray(
        [
            2.58520000e00,
            2.58520000e01,
            8.01412000e01,
            2.74031200e02,
            8.11752800e02,
            2.69636360e03,
            8.22093600e03,
            2.72066448e04,
            8.03764532e04,
            2.72402524e05,
            5.29924637e05,
            1.32317775e06,
            2.64755244e06,
        ]
    )
    q_array = np.asarray(
        [
            36272,
            34309,
            32760,
            29308,
            25860,
            23927,
            25416,
            30701,
            36748, 
            46590,
            52648,
            61358,
            66656,
        ]
    )

    for p in range(13):
        ########################################
        ### 1. Plasma parameters in SI units ###
        ########################################

        T_e = 30000  # electron temperature (K)
        T_n = 300  # neutral gas temperature (K)
        n_inf = 3.57167962497e15  # ion concentration (1/m^3)
        m_i = 1.673557e-27  # H+-ion mass (kg)
        w_c = w_c_array[p]  # ion-neutral collision frequency (s^-1)
        E = E_array[p]  # electric field (V/m)

        distributionType = "fieldDriven"
        plasmaParametersInSIUnits = PlasmaParametersInSIUnitsFieldDriven(
            T_n, T_e, n_inf, E, w_c, m_i
        )
        plasmaParametersInSIUnits.printParameters()

        ############################################
        ### 2. Simulation parameters in SI units ###
        ############################################

        R = 1.25 * plasmaParametersInSIUnits.r_D_e
        H = 6 * plasmaParametersInSIUnits.r_D_e
        N = int(2 ** 17)
        n = 200000
        d_t = 1e-11
        simulationParametersInSIUnits = SimulatioParametersInSIUnits(
            R, H, N, n, d_t, plasmaParametersInSIUnits
        )
        simulationParametersInSIUnits.printParameters()

        ############################
        ### 3. Output parameters ###
        ############################

        nOutput = 100000
        nFileOutput = 100000
        csvOutputFileName = ""
        xyzOutputFileName = (
            "/home/avtimofeev/kolotinskii/opendust/data/Patacchini2008/Figure3/{}point/trajectory.xyz".format(
                p + 1
            )
        )
        restartFileName = (
            "/home/avtimofeev/kolotinskii/opendust/data/Patacchini2008/Figure3/{}point/RESTART".format(p + 1)
        )
        outputParameters = OutputParameters(
            nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
        )
        #########################
        ### 4. Dust particles ###
        #########################

        r = 1e-05  # radius of dust particles
        q = q_array[p] * plasmaParametersInSIUnits.e  # charge of dust particles
        chargeCalculationMethod = "given"  # charge calculation method

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

        forceDust1 = openDust.dustParticles[0].forceDust
        forceIonsCollect1 = openDust.dustParticles[0].forceIonsCollect
        forceIonsOrbit1 = openDust.dustParticles[0].forceIonsOrbit
        forceExternalField = openDust.dustParticles[0].forceExternalField
        q1 = openDust.dustParticles[0].q
        t = openDust.t

        f = open(
            "/home/avtimofeev/kolotinskii/opendust/data/Patacchini2008/Figure3/{}point/force.txt".format(
                p + 1
            ),
            "w",
        )
        for i in range(n):
            f.write(
                "{}\t{}\t{}\t{}\t{}\n".format(
                    t[i],
                    forceIonsOrbit1[i][2],
                    forceIonsCollect1[i][2],
                    forceDust1[i][2],
                    forceExternalField[i][2],
                )
            )
        f.close()

        meanForceIonsOrbit1 = 0
        meanForceIonsCollect1 = 0
        meanForceField1 = 0
        meanCharge1 = 0
        sigmaForceIonsOrbit1 = 0
        sigmaForceIonsCollect1 = 0
        sigmaForceField1 = 0
        sigmaCharge1 = 0
        iterator = 0
        for i in range(n):
            if i > 100000:
                iterator += 1
                meanForceIonsOrbit1 += forceIonsOrbit1[i][2]
                meanForceIonsCollect1 += forceIonsCollect1[i][2]
                meanForceField1 += forceExternalField[i][2]
                meanCharge1 += q1[i]
        meanForceIonsOrbit1 /= iterator
        meanForceIonsCollect1 /= iterator
        meanForceField1 /= iterator
        meanCharge1 /= iterator
        iterator = 0
        for i in range(n):
            if i > 100000:
                iterator += 1
                sigmaForceIonsOrbit1 += (
                    forceIonsOrbit1[i][2] - meanForceIonsOrbit1
                ) ** 2
                sigmaForceIonsCollect1 += (
                    forceIonsCollect1[i][2] - meanForceIonsCollect1
                ) ** 2
                sigmaForceField1 += (forceExternalField[i][2] - meanForceField1) ** 2
                sigmaCharge1 += (q1[i] - meanCharge1) ** 2
        sigmaForceIonsOrbit1 = math.sqrt(sigmaForceIonsOrbit1 / iterator)
        sigmaForceIonsCollect1 = math.sqrt(sigmaForceIonsCollect1 / iterator)
        sigmaForceField1 = math.sqrt(sigmaForceField1 / iterator)
        sigmaCharge1 = math.sqrt(sigmaCharge1 / iterator)

        print(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                w_c,
                meanForceIonsOrbit1,
                meanForceIonsCollect1,
                meanForceField1,
                meanCharge1,
                sigmaForceIonsOrbit1,
                sigmaForceIonsCollect1,
                sigmaForceField1,
                sigmaCharge1,
            )
        )

        f = open(
            "/home/avtimofeev/kolotinskii/opendust/data/Patacchini2008/Figure3/{}point/charge.txt".format(
                p + 1
            ),
            "w",
        )
        for i in range(n):
            f.write("{}\t{}\n".format(t[i], q1[i]))
        f.close()

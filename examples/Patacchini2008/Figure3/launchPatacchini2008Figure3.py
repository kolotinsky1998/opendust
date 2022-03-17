from opendust.opendust import DustParticle
from opendust.opendust import PlasmaParametersInSIUnitsFieldDriven
from opendust.opendust import SimulatioParametersInSIUnits
from opendust.opendust import OutputParameters
from opendust.opendust import OpenDust

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
        ###############################################
        ### 1. Define plasma parameters in SI units ###
        ###############################################

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

        ###################################################
        ### 2. Define simulation parameters in SI units ###
        ###################################################

        R = 3 * plasmaParametersInSIUnits.r_D_e
        H = 6 * plasmaParametersInSIUnits.r_D_e
        N = int(2 ** 17)
        n = 200000
        d_t = 1e-11
        simulationParametersInSIUnits = SimulatioParametersInSIUnits(
            R, H, N, n, d_t, plasmaParametersInSIUnits
        )
        simulationParametersInSIUnits.printParameters()

        ###################################
        ### 3. Define output parameters ###
        ###################################

        directory = "/home/avtimofeev/opendust/data/Patacchini2008/Figure3/{}point/".format(p + 1)
        nOutput = 49999
        nFileOutput = 49999
        csvOutputFileName = directory + "csv/trajectory"
        xyzOutputFileName = directory + "trajectory.xyz"
        restartFileName = directory + "RESTART"
        outputParameters = OutputParameters(
            nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
        )

        ################################
        ### 4. Define dust particles ###
        ################################

        r = 1e-05  # radius of dust particles
        q = 0 * plasmaParametersInSIUnits.e  # charge of dust particles
        chargeCalculationMethod = "consistent"  # charge calculation method

        x_1, y_1, z_1, r_1, q_1 = 0, 0, -1 * plasmaParametersInSIUnits.r_D_e, r, q

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

        openDust.simulate(deviceIndex = "0,1,2,3,4,5,6,7", cutOff = False)


        ##################
        ### 6. Analyze ###
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
        meanCharge = 0

        iterator = 0

        for i in range(n):
            if i > 150000:
                iterator += 1
                meanForceIonsOrbitZ += forceIonsOrbitZ[i][2]
                meanForceIonsCollectZ += forceIonsCollectZ[i][2]
                meanCharge += q[i]

        meanForceIonsOrbitZ /= iterator
        meanForceIonsCollectZ /= iterator
        meanCharge /= iterator

        file = open("force.txt",'a')
        file.write("{}\t{}\t{}\n".format(w_c, meanForceIonsOrbitZ, meanForceIonsCollectZ))
        file.close()
        
        file = open("charge.txt",'a')
        file.write("{}\t{}\n".format(w_c, meanCharge))
        file.close()


        f = open(directory+"charge.txt","w")

        for i in range(n):
            f.write(
                "{}\t{}\n".format(
                    t[i],
                    q[i]
                )
            )
        f.close()
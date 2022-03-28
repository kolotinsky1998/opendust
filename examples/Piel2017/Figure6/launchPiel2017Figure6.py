from opendust.opendust import DustParticle
from opendust.opendust import PlasmaParametersInSIUnitsMaxwell
from opendust.opendust import SimulatioParametersInSIUnits
from opendust.opendust import OutputParameters
from opendust.opendust import OpenDust

import numpy as np

if __name__ == "__main__":

    M_array = np.asarray(
        [
            0.29940,
            0.40129,
            0.50033,
            0.60133,
            0.70098,
            0.80019,
            1.00091,
        ]
    )
    for p in range(7):
        ###############################################
        ### 1. Define plasma parameters in SI units ###
        ###############################################

        T_e = 29011  # electron temperature (K)
        T_n = 290.11  # neutral gas temperature (K)
        n_inf = 1e14  # ion concentration (1/m^3)
        m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
        M = M_array[p]  # Mach number of the ion flow

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
        n = 50000
        d_t = 1e-09
        simulationParametersInSIUnits = SimulatioParametersInSIUnits(
            R, H, N, n, d_t, plasmaParametersInSIUnits
        )
        simulationParametersInSIUnits.printParameters()

        ###################################
        ### 3. Define output parameters ###
        ###################################

        directory = "/home/avtimofeev/opendust/data/Piel2017/Figure6/point{}/".format(p + 1)
        nOutput = 10000
        nFileOutput = 10000
        csvOutputFileName = directory + "csv/trajectory"
        xyzOutputFileName = directory + "trajectory.xyz"
        restartFileName = directory + "RESTART"
        outputParameters = OutputParameters(
            nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
        )

        ################################
        ### 4. Define dust particles ###
        ################################
        
        r = 12.0e-6  # radius of dust particles
        q = 40939.491598 * plasmaParametersInSIUnits.e  # charge of dust particles
        chargeCalculationMethod = "given"  # charge calculation method

        x_1, y_1, z_1, r_1, q_1 = 0, 0, -0.5 * plasmaParametersInSIUnits.r_D_e, r, q
        x_2, y_2, z_2, r_2, q_2 = (
            0.5 * plasmaParametersInSIUnits.r_D_e,
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

        openDust.simulate(deviceIndex = "0,1", cutOff = False)

        ##################
        ### 7. Analyze ###
        ##################

        forceIonsOrbit1 = openDust.dustParticles[0].forceIonsOrbit
        forceDust1 = openDust.dustParticles[0].forceDust
        forceIonsOrbit2 = openDust.dustParticles[1].forceIonsOrbit
        forceDust2 = openDust.dustParticles[1].forceDust
        t = openDust.t
        q = openDust.dustParticles[0].q

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

        f = open(directory+"charge.txt","w")

        for i in range(n):
            f.write(
                "{}\t{}\n".format(
                    t[i],
                    q[i]
                )
            )
        f.close()

        meanForce1X = 0
        meanForce1Z = 0
        meanForce2X = 0
        meanForce2Z = 0

        iterator = 0

        for i in range(n):
            if i > 10000:
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
                M_array[p], meanForce1X, meanForce1Z, meanForce2X, meanForce2Z,
            )
        )


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

    ########################################
    ### 1. Plasma parameters in SI units ###
    ########################################

    T_e = 29011  # electron temperature (K)
    T_n = 290.11  # neutral gas temperature (K)
    n_inf = 1e14  # ion concentration (1/m^3)
    m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
    M = 1 #Mach number of the ion flow

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
    n = 3000
    d_t = 3.5148240854e-09
    simulationParametersInSIUnits = SimulatioParametersInSIUnits(
        R, H, N, n, d_t, plasmaParametersInSIUnits
    )
    simulationParametersInSIUnits.printParameters()

    ############################
    ### 3. Output parameters ###
    ############################

    directory = "/home/avtimofeev/kolotinskii/opendust/data/Piel2017/Figure1/oml/"
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
    q = 0 * plasmaParametersInSIUnits.e  # charge of dust particles
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

    sigmaForceIonsOrbitZ = 0
    sigmaForceIonsCollectZ = 0
    iterator = 0

    for i in range(n):
        if i > 1000:
            iterator += 1
            meanForceIonsOrbitZ += forceIonsOrbitZ[i][2]
            meanForceIonsCollectZ += forceIonsCollectZ[i][2]

    meanForceIonsOrbitZ /= iterator
    meanForceIonsCollectZ /= iterator

    iterator = 0
    for i in range(n):
        if i > 2000:
            iterator += 1
            sigmaForceIonsOrbitZ += (
                forceIonsOrbitZ[i][2] - meanForceIonsOrbitZ
            ) ** 2
            sigmaForceIonsCollectZ += (
                forceIonsCollectZ[i][2] - meanForceIonsCollectZ
            ) ** 2

    sigmaForceIonsOrbitZ = math.sqrt(sigmaForceIonsOrbitZ / iterator)
    sigmaForceIonsCollectZ = math.sqrt(sigmaForceIonsCollectZ / iterator)


    print("Mean force from ions orbits = {}".format(meanForceIonsOrbitZ))
    print("Mean force from collected ions = {}".format(meanForceIonsCollectZ))
    print("Mean force from ions = {}".format(meanForceIonsOrbitZ+meanForceIonsCollectZ))
    print("Fluctuation of force from ions orbits = {}".format(sigmaForceIonsOrbitZ))
    print("Fluctuation of force from collected ions = {}".format(sigmaForceIonsCollectZ))

    f = open(directory+"charge.txt","w")

    for i in range(n):
        f.write(
            "{}\t{}\n".format(
                t[i],
                q[i]
            )
        )
    f.close()

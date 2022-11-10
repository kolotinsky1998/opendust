#####################################
### 0. Load all necessary modules ###
#####################################

from opendust.opendust import DustParticle
from opendust.opendust import PlasmaParametersInSIUnitsMaxwell
from opendust.opendust import SimulationParametersInSIUnits
from opendust.opendust import OutputParameters
from opendust.opendust import OpenDust


###############################################
### 1. Define plasma parameters in SI units ###
###############################################

T_e = 29011  # electron temperature (K)
T_i = 290.11  # ion temperature (K)
n_inf = 1e14  # ion concentration (1/m^3)
m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
M = 1  # Mach number of the ion flow

distributionType = "Maxwellian"
plasmaParametersInSIUnits = PlasmaParametersInSIUnitsMaxwell(
    T_i, T_e, n_inf, M, m_i
)
plasmaParametersInSIUnits.printParameters()

###################################################
### 2. Define simulation parameters in SI units ###
###################################################

R = 3 * plasmaParametersInSIUnits.r_D_e
H = 6 * plasmaParametersInSIUnits.r_D_e
N = int(2 ** 16)
n = 3000
d_t = 3.5148240854e-09
simulationParametersInSIUnits = SimulationParametersInSIUnits(
    R, H, N, n, d_t, plasmaParametersInSIUnits
)
simulationParametersInSIUnits.printParameters()

###################################
### 3. Define output parameters ###
###################################

directory = "./"
nOutput = 100
nFileOutput = 100
csvOutputFileName = directory + "csv/trajectory"
xyzOutputFileName = directory + "trajectory.xyz"
restartFileName = directory + "RESTART"
outputParameters = OutputParameters(
    nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
)

################################
### 4. Define dust particles ###
################################

r = 58.8e-6  # radius of dust particles
q = 390000 * plasmaParametersInSIUnits.e  # charge of dust particles
chargeCalculationMethod = "given"  # charge calculation method

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


openDust.simulate()

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

f = open(directory+"charge.txt","w")

for i in range(n):
    f.write(
        "{}\t{}\n".format(
            t[i],
            q[i]
        )
    )
f.close()

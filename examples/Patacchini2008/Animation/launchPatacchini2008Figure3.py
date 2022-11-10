from opendust.opendust import DustParticle
from opendust.opendust import PlasmaParametersInSIUnitsFieldDriven
from opendust.opendust import SimulationParametersInSIUnits
from opendust.opendust import OutputParameters
from opendust.opendust import OpenDust


###############################################
### 1. Define plasma parameters in SI units ###
###############################################

T_e = 30000  # electron temperature (K)
T_n = 300  # neutral gas temperature (K)
n_inf = 3.57167962497e15  # ion concentration (1/m^3)
m_i = 1.673557e-27  # H+-ion mass (kg)
w_c = 1.65562835e08  # ion-neutral collision frequency (s^-1)
E = 2.72066448e04  # electric field (V/m)

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
N = int(2 ** 19)
n = 50000
d_t = 1e-11
simulationParametersInSIUnits = SimulationParametersInSIUnits(
    R, H, N, n, d_t, plasmaParametersInSIUnits
)
simulationParametersInSIUnits.printParameters()

###################################
### 3. Define output parameters ###
###################################

directory = "/home/avtimofeev/opendust/data/Patacchini2008/Animation/"
nOutput = 500
nFileOutput = 500
csvOutputFileName = directory + "csv/trajectory"
xyzOutputFileName = directory + "trajectory.xyz"
restartFileName = directory + "RESTART"
outputParameters = OutputParameters(
    nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
)

################################
### 4. Define dust particles ###
################################

r = 1e-05  # radius of dust particles (m)
q = 200000 * plasmaParametersInSIUnits.e  # charge of dust particles
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

openDust.simulate(deviceIndex = "0,1,2", cutOff = False)

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

iterator = 0

for i in range(n):
    if i > 25000:
        iterator += 1
        meanForceIonsOrbitZ += forceIonsOrbitZ[i][2]
        meanForceIonsCollectZ += forceIonsCollectZ[i][2]

meanForceIonsOrbitZ /= iterator
meanForceIonsCollectZ /= iterator


print("Mean force from ions orbits = {}".format(meanForceIonsOrbitZ))
print("Mean force from collected ions = {}".format(meanForceIonsCollectZ))
print("Mean force from ions = {}".format(meanForceIonsOrbitZ+meanForceIonsCollectZ))


f = open(directory+"charge.txt","w")

for i in range(n):
    f.write(
        "{}\t{}\n".format(
            t[i],
            q[i]
        )
    )
f.close()

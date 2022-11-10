# Tutorial

## 1. Example Files Overview

Currently, only one example file is provided in the tutorial folder. More examples will be added soon.

* **HelloMaxwellian** A very simple example intended for verifying that you have installed OpenDust correctly. 
It also introduces you to the basic classes within OpenDust and shows how to launch the simulation of the
Maxwellian plasma flow around a single dust particle.


## 2. **HelloMaxwellian**
In this example, the simulation of the collisionless Maxwellian plasma flow around a solitary dust particle with a preset charge is performed.
Forces, acting on the dust particle, are recorded for every integration time step. 


Go to the HelloMaxwellian directory and open launch.py file. You can launch the simulation simply type ```python launch.py```. That's it! 

This Python "launch.py" file is the only thing needed to run the simulation within OpenDust. The file is devided into six logicaly steps with the additional zero step. 

On the zero step you should load all OpenDust and any additional Python modules you will use bellow. 
Necessary OpenDust modules for this simulation are

``` python
from opendust.opendust import DustParticle
from opendust.opendust import PlasmaParametersInSIUnitsMaxwell
from opendust.opendust import SimulationParametersInSIUnits
from opendust.opendust import OutputParameters
from opendust.opendust import OpenDust

```

1. **Define plasma parameters in SI units.** In the first step, user should define plasma parameters of the simulation, namely: electron temperature 
```T_e```, ion temperature ```T_i```, ion concentrations in the unperturbed plasma region ```n_inf```, ion mass ```m_i```, and Mach number of the ion flow ```M``` corresponding to the plasma Bohm velocity. All these parameters should be given in SI units and are used to create ```PlasmaParametersInSIUnitsMaxwell``` class object. This class has ```printParameters()``` method which prints plasma parameters to a console in the beginning of the simulation. 

``` Python 
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
```
2. **Define simulation parameters in SI units.** In the second step, simulation parameters are defined: radius ```R`` and height ```H``` 
of the cylindrical computational domain, number of discrete particle (ion clouds) used in simulation ```N```, number of integration time steps ```d_t```.
For convinience, radius ```R`` and height ```H``` of the cylindrical computational domain can be defined with the help of ```plasmaParametersInSIUnits.r_D_e``` attribute, which is the electron Debye radius. These parameters are used to create ```SimulationParametersInSIUnits``` class object. This class has ```printParameters()``` method which prints simulation parameters to a console in the beginning of the simulation.

``` Python
R = 3 * plasmaParametersInSIUnits.r_D_e
H = 6 * plasmaParametersInSIUnits.r_D_e
N = int(2 ** 16)
n = 3000
d_t = 3.5148240854e-09
simulationParametersInSIUnits = SimulationParametersInSIUnits(
    R, H, N, n, d_t, plasmaParametersInSIUnits
)
simulationParametersInSIUnits.printParameters()
```
3. **Define output parameters.*** In the third step, output parameters are defined: a frequency with that the simulation information is printed to the console ```nOutput```, a frequency with that positions of ions are recorded to the output files ```nFileOutput```, absolute path to the .xyz file, in which positions of ions are recorded during simulation ```csvOutputFileName```, absolute path to the basename of .csv files, in which positions of ions are recorded during simulation ```xyzOutputFileName```, absolute path to the file, in which whole state of the system will be recorded at the end of the simulation ```restartFileName```. ```restartFileName``` can be used to restart simulation from the recorded state. These parameters are used to create ```OutputParameters``` class object. 

``` Python
directory = "./"
nOutput = 100
nFileOutput = 100
csvOutputFileName = directory + "csv/trajectory"
xyzOutputFileName = directory + "trajectory.xyz"
restartFileName = directory + "RESTART"
outputParameters = OutputParameters(
    nOutput, nFileOutput, csvOutputFileName, xyzOutputFileName, restartFileName
)
```
4. **Define dust particles.** In the fourth step, dust particles parameter should be defined, namely: radius of each dust particles ```r_1``` (here only one), initial charge of dust particles ```q_1``` (here only one), calculation method of dust particles charges ```chargeCalculationMethod``` (here preset option is chosen with ```given``` value), positions of dust particles ```x_1, y_1, z_1``` (here only one). These parameters are the used to create ```DustParticle``` dust particle class objects for each dust particles (here only one). After that, the Python list of the created ```DustParticle``` class objects should be constructed.
```Python 
r = 58.8e-6  # radius of dust particles
q = 390000 * plasmaParametersInSIUnits.e  # charge of dust particles
chargeCalculationMethod = "given"  # charge calculation method

x_1, y_1, z_1, r_1, q_1 = 0, 0, -1 * plasmaParametersInSIUnits.r_D_e, r, q

dustParticle1 = DustParticle(x_1, y_1, z_1, r_1, chargeCalculationMethod, q_1)

dustParticles = [dustParticle1]
```
5. **Create OpenDust class object and start simulation.** In the fifth step, all defined bellow objects are used to create the main ```OpenDust``` class object, which contains all the simulation information and ```simulate()``` method to launch the simulation. 

```Python
openDust = OpenDust(
    plasmaParametersInSIUnits,
    simulationParametersInSIUnits,
    outputParameters,
    dustParticles,
    distributionType,
)


openDust.simulate()
```

5. **Analyze.** In the sixth step, user can add any code to analyze the simulated system using the main member of the ```OpenDust``` class, namely, the ```dustParticles``` object. ```dustParticles``` is a Python list which contain the separate object of each dust particle defined in the simulation. This example contains only one dust particle, which object is available via ```dustParticles[0]```. This dust particle object stores arrays of charges ```q```, forces from ion-dust electric interactions ```forceIonsOrbit``` and forces from momentum transfer in direct ion-dust collisions ```forceIonsCollect``` in each discrete simulation time step. ```forceIonsOrbit``` and ```forceIonsCollect``` objects are arrays with length 3, first, second and third elements of that are force components on three coordinate axes x, y, z correspondingly. ```OpenDust``` class object contains one more member which may be useful for analysis. It is an array of all time moments of the simulation ```t```. In this example, described objecgs are used to write down time-dependent charge and force to "charge.txt" and "force.txt" files correspondingly. 

```Python
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
```

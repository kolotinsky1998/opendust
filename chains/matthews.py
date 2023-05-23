import numpy as np
from numpy import random
import cupy as cp
from unyt import nm
from unyt import ps
from unyt import m
from unyt import s
from unyt import J
from unyt import kJ
from unyt import kg
from unyt import amu
import openmm as mm
from scipy import special
import csv
from scipy import integrate
from scipy.interpolate import UnivariateSpline
from sympy import diff
from sympy import symbols


""" DustParticle class

stores information about a dust particle 
"""


class DustParticle:
    def __init__(self, x, y, z, r, chargeCalculationMethod, q=0.0):
        """! Initialize DustParticle class
        @param x    x-coordinate of a dust particle
        @param y    y-coordinate of a dust particle
        @param z    z-coordinate of a dust particle
        @param r    radius of a dust particle
        @param q    charge of a dust particle
        @param chargeCalculationMethod     description of how charge of a dust particle is calculated (string)
        """
        self.x = x
        self.y = y
        self.z = z
        self.chargeCalculationMethod = chargeCalculationMethod
        self.q = q
        self.r = r


""" PlasmaParametersInSIUnitsMatthews class 

stores plasma parameters in SI units in case of matthews method
"""

class PlasmaParametersInSIUnitsMatthews:
    def __init__(self, T_n, T_e, n_inf, externalFieldOrder, Phi_parameters, M, w_c, m_i):
        """! Initialize PlasmaParametersInSIUnitsMatthews class
        @param T_n      neutral gas temperature
        @param T_e      electron temperature
        @param n_inf    unperturbed density of the plasma in the quasineutral plasma region
        @param Phi_expression        expression for the external electric field along z-axis
        @param Phi_parameters        parameters for the external electric field along z-axis
        @param M        Mach number of the ions entering the simulation domain 
        @param w_c      ion-neutral collision frequency
        @param m_i      mass of the ion
        """
        self.T_n = T_n
        self.T_e = T_e
        self.n_inf = n_inf
        self.externalFieldOrder = externalFieldOrder
        self.Phi_parameters = Phi_parameters
        self.M = M
        self.m_i = m_i

        self.k_B = 1.380649e-23
        self.e = -1.60217662e-19
        self.eps_0 = 8.85418781762039e-12
        self.m_e = 9.0938356e-31

        self.r_D_e = np.sqrt(
            (self.eps_0 * self.k_B * self.T_e) / (self.e ** 2 * self.n_inf)
        )
        self.v_B = np.sqrt(self.k_B * self.T_e / self.m_i)
        self.w_p_e = np.sqrt((self.n_inf * self.e ** 2) / (self.m_e * self.eps_0))
        self.w_p_i = np.sqrt((self.n_inf * self.e ** 2) / (self.m_i * self.eps_0))
        self.w_c = w_c
        self.v_T = np.sqrt(self.k_B * self.T_n / self.m_i)

        self.v_fl = M  * self.v_B

    def printParameters(self):
        """! Print parameters to a console"""
        print("#########################################")
        print("##### Plasma parameters in SI units #####")
        print("#########################################")
        print("\n")

        print("Neutral gas temperature T_n = {} K".format(self.T_n))
        print("Electron temperature T_e = {} K".format(self.T_e))
        print(
            "Unperturbed density of the plasma in the quasineutral plasma region n_inf = {} 1/m**3".format(
                self.n_inf
            )
        )
        print("Mach number of the ions entering the simulation domain  M = {}".format(self.M))
        print("Ion mass m_i = {} kg".format(self.m_i))

        print("Electron Debye radius r_D_e = {} m".format(self.r_D_e))
        print("Bohm velocity v_B = {} m/s".format(self.v_B))
        print("Electron plasma frequency w_p_e = {} radian/s".format(self.w_p_e))
        print("Ion plasma frequency w_p_i = {} radian/s".format(self.w_p_i))
        print("Thermal neutral gas velocity v_T = {} m/s".format(self.v_T))
        print("Ion flow velocity v_fl = {} m/s".format(self.v_fl))
        print("Ion-neutral collision frequency w_c = {} 1/s".format(self.w_c))
        print("Order of external electric field: {}".format(self.externalFieldOrder))
        for key in self.Phi_parameters:
            print("{} = {} ({})".format(key,self.Phi_parameters[key][0], self.Phi_parameters[key][1]))
        print("\n")


""" SimulatioParametersInSIUnits class 

stores information about simulation paremeters such as
intergration step, radius of a spherical computational domain, etc.
"""


class SimulatioParametersInSIUnits:
    def __init__(self, R, H, N, n, d_t, plasmaParametersInSIUnits):
        """! Initialize SimulatioParametersInSIUnits class
        @param R        radius of a cylindrical simulation domain
        @param H        height of a cylindrical simulation domain
        @param N        number of superions used in simulation
        @param n        number of integration time steps
        @param plasmaParametersInSIUnits    plasma parameters in SI units
        """
        n_inf = plasmaParametersInSIUnits.n_inf
        r_D_e = plasmaParametersInSIUnits.r_D_e

        self.R = R
        self.H = H
        self.N = N
        self.n = n
        self.N_s = np.pi * R ** 2 * H * n_inf / self.N

        self.d_t = d_t
        self.r_0 = r_D_e * 1.0e-3

    def printParameters(self):
        """! Print plasma parameters to a console"""
        print("#########################################")
        print("### Simulation parameters in SI units ###")
        print("#########################################")
        print("\n")

        print("Radius of the cylindrical computational domain R = {} m".format(self.R))
        print("Height of the cylindrical computational domain H = {} m".format(self.H))
        print("Number of superions N = {}".format(self.N))
        print("Number of ions in one superion N_s = {}".format(self.N_s))
        print("Integration time step d_t = {} s".format(self.d_t))
        print("Number of time steps n = {}".format(self.n))
        print("\n")


""" OutputParameters class

stores information about output parameters such as
output file names and output frequency.
"""


class OutputParameters:
    def __init__(
        self,
        nOutput=10,
        nFileOutput=10,
        csvOutputFileName="",
        xyzOutputFileName="",
        restartFileName="",
        trapFileName="",
    ):
        """! Initialize SimulatioParametersInSIUnits class
        @param nOutput      each nOutput step general information is printed to a console
        @param nFileOutput  each nFileOutput step coordinates of superions are written to a file
        @param csvOutputFileName    basename of the CSV-format files where coordinates of superions are recorded
        @param xyzOutputFileName    name of the XYZ-format file where coordinates of superions are recorded
        @param restartFileName      name of the restart file where ion system state is written
        @param trapFileName         name of the file where radial trap coefficients are written
        """
        self.nOutput = nOutput
        self.nFileOutput = nFileOutput
        self.csvOutputFileName = csvOutputFileName
        self.xyzOutputFileName = xyzOutputFileName
        self.restartFileName = restartFileName
        self.trapFileName = trapFileName


""" OpenDust class 

is the main class of the openDust package.
"""


class OpenDust:
    def __init__(
        self,
        plasmaParametersInSIUnits,
        simulationParametersInSIUnits,
        outputParameters,
        dustParticles,
    ):
        """! Initialize SimulationParametersInSIUnits class
        @param plasmaParametersInSIUnits      stores plasma parameters in SI units
        @param simulationParametersInSIUnits  stores simulation parameters in SI units
        @param outputParameters               stores output parameters
        @param dustParticles                  stores list of dust particles used in simulation
        """
        self.plasmaParametersInSIUnits = plasmaParametersInSIUnits
        self.simulationParametersInSIUnits = simulationParametersInSIUnits
        self.outputParameters = outputParameters

        """ Get simulation parameters and convert them to OpenMM units """
        self.R = float((simulationParametersInSIUnits.R * m).to("nm").value)
        self.H = float((simulationParametersInSIUnits.H * m).to("nm").value)
        self.d_t = float((simulationParametersInSIUnits.d_t * s).to("ps").value)
        self.r_0 = float((simulationParametersInSIUnits.r_0 * m).to("nm").value)
        self.N = simulationParametersInSIUnits.N
        self.N_s = simulationParametersInSIUnits.N_s
        self.n = simulationParametersInSIUnits.n

        """ Get some fundamental constants from plasmaParametersInSIUnits """
        self.k = 1.0 / (plasmaParametersInSIUnits.eps_0 * 4.0 * np.pi)
        self.eps_0 = plasmaParametersInSIUnits.eps_0
        self.e = plasmaParametersInSIUnits.e

        """ Get plasma parameters and convert needed to OpenMM units """
        self.n_inf = float(
            (plasmaParametersInSIUnits.n_inf / m ** 3).to("1/nm**3").value
        )
        self.r_D_e = float((plasmaParametersInSIUnits.r_D_e * m).to("nm").value)
        self.v_fl = float((plasmaParametersInSIUnits.v_fl * m / s).to("nm/ps").value)
        self.v_T = float((plasmaParametersInSIUnits.v_T * m / s).to("nm/ps").value)
        self.q_s = abs(plasmaParametersInSIUnits.e) * self.N_s
        self.m_s = float(
            (plasmaParametersInSIUnits.m_i * self.N_s * kg).to("amu").value
        )

        self.Phi_parameters = plasmaParametersInSIUnits.Phi_parameters
        self.externalFieldOrder = plasmaParametersInSIUnits.externalFieldOrder
   
        self.w_c = float((plasmaParametersInSIUnits.w_c / s).to("1/ps").value)

        """ Define conversion coefficient to calculate forces in OpenMM units """
        self.interactionConversionCoefficient = (1 * J).to("kJ/mol").value * (1 * m).to(
            "nm"
        ).value
        self.externalFieldConversionCoefficient = (1 * J).to("kJ/mol").value * (
            1 * nm
        ).to("m").value

        """ Define time array in SI units """
        self.t = np.linspace(0, self.n * simulationParametersInSIUnits.d_t, self.n)

        """ Create local class to store dust particle properties """

        class DustParticle:
            def __init__(self, dustParticle, n):
                self.x = float((dustParticle.x * m).to("nm").value)
                self.y = float((dustParticle.y * m).to("nm").value)
                self.z = float((dustParticle.z * m).to("nm").value)
                self.r = float((dustParticle.r * m).to("nm").value)

                self.q = np.zeros(n)
                for i in range(n):
                    self.q[i] = dustParticle.q
                self.chargeCalculationMethod = dustParticle.chargeCalculationMethod
                """ Force from direct collisions between ions and dust particle """
                self.forceIonsCollect = np.zeros((n, 3))
                """ Force from ions without direct collisions """
                self.forceIonsOrbit = np.zeros((n, 3))
                """ Force from other dust particles """
                self.forceDust = np.zeros((n, 3))
                """ Force from external electric field """
                self.forceExternalField = np.zeros((n, 3))

        """ Define dust particles """
        self.dustParticles = []
        self.numberOfDustParticles = len(dustParticles)
        for p in range(self.numberOfDustParticles):
            self.dustParticles.append(
                DustParticle(dustParticles[p], self.n)
            )
        self.dustParticleForceLines = []

        """ Define strings for OpennMM force calculation from dust particles """
        for p in range(self.numberOfDustParticles):
            _dustParticleForceLine = "q" + str(p) + "*k*q_s/r; "
            _dustParticleForceLine += "r=sqrt((x-x_p)^2+(y-y_p)^2+(z-z_p)^2); "
            _dustParticleForceLine += "k={}; q_s={}; x_p={}; y_p={}; z_p={}".format(
                self.k * self.interactionConversionCoefficient,
                self.q_s,
                self.dustParticles[p].x,
                self.dustParticles[p].y,
                self.dustParticles[p].z,
            )
            self.dustParticleForceLines.append(_dustParticleForceLine)

    def simulate(self, deviceIndex = "0", cutOff = False, toRestartFileName="", considerTrap = True, trapFileName=""):
        """! Conduct simulation
        @param palatformName    name of the platform used for calculation
        @param restartFileName  name of the restart file to read ion positions and velocities
        """
        
        platformName="CUDA"

        position = []
        velocity = []

        if toRestartFileName == "":
            """Generate positions and velocities according to distribution"""

            for i in range(self.N):
                flag = True
                while flag:
                    r = np.sqrt(random.uniform(0, 1))
                    theta = random.uniform(0, 1)
                    x = self.R * r * np.cos(2 * np.pi * theta)
                    y = self.R * r * np.sin(2 * np.pi * theta)
                    z = random.uniform(-self.H / 2, self.H / 2)

                    r_pr = np.zeros(self.numberOfDustParticles)
                    for j in range(self.numberOfDustParticles):
                        r_pr[j] = np.sqrt(
                            (x - self.dustParticles[j].x) ** 2
                            + (y - self.dustParticles[j].y) ** 2
                            + (z - self.dustParticles[j].z) ** 2
                        )
                    outsideTheDustParticles = True
                    for j in range(self.numberOfDustParticles):
                        if r_pr[j] <= self.dustParticles[j].r:
                            outsideTheDustParticles = False
                            break
                    if outsideTheDustParticles:
                        position.append((x, y, z))
                        flag = False

            v_x = np.random.normal(0, self.v_T, self.N)
            v_y = np.random.normal(0, self.v_T, self.N)
            v_z = np.random.normal(self.v_fl, self.v_T, self.N)
            
            for i in range(self.N):
                velocity.append((v_x[i], v_y[i], v_z[i]))
        else:
            """Read positions and velocities from restart file"""
            toRestartFile = open(toRestartFileName, "r")
            for i in range(self.N):
                _line = toRestartFile.readline()
                _line_attribute = _line.split("\t")
                x = float(_line_attribute[0])
                y = float(_line_attribute[1])
                z = float(_line_attribute[2])
                v_x = float(_line_attribute[3])
                v_y = float(_line_attribute[4])
                v_z = float(_line_attribute[5])
                position.append((x, y, z))
                velocity.append((v_x, v_y, v_z))
            toRestartFile.close()

        """ Create system """
        system = mm.System()

        """ Define interions forces """
        ionIonForceLine = "kq_sq_s*exp(-r/r_D_e)/(r+r_0)"
        ionIonForce = mm.CustomNonbondedForce(ionIonForceLine)
        ionIonForce.addGlobalParameter(
            "kq_sq_s",
            self.k * self.interactionConversionCoefficient * self.q_s * self.q_s,
        )
        ionIonForce.addGlobalParameter("r_D_e", self.r_D_e)
        ionIonForce.addGlobalParameter("r_0", self.r_0)
        ionIonForce.setForceGroup(1)

        if cutOff == True:
            ionIonForce.setNonbondedMethod(1)
            ionIonForce.setCutoffDistance(3.2 * self.r_D_e)
            ionIonForce.setUseSwitchingFunction(True)
            ionIonForce.setSwitchingDistance(3 * self.r_D_e)

        """ Define ion-dust particle forces """
        dustParticleForces = []
        for p in range(self.numberOfDustParticles):
            dustParticleForces.append(
                mm.CustomExternalForce(self.dustParticleForceLines[p])
            )
            dustParticleForces[p].addGlobalParameter(
                "q" + str(p), self.dustParticles[p].q[0]
            )

        """ Define ion background otside domain forces """
        if considerTrap:
            def computeAndFitBackgroundPotential(R, H, r_D_e, N, position):
                """Approximation of ion background potential with 8th degree even polinomial"""
                R = R / r_D_e
                H = H / r_D_e
                N_R = int(R * 10)
                N_H = int(H * 10)
                r_0 = 0.001
                grid_R_gpu = cp.linspace(0, R, N_R)
                grid_H_gpu = cp.linspace(-H / 2.0, H / 2.0, N_H)
                potential_gpu = cp.zeros((N_H, N_R))

                x = cp.zeros(N)
                y = cp.zeros(N)
                z = cp.zeros(N)
                
                for p in range(N):
                    x[p], y[p], z[p] = position[p][0] / r_D_e, position[p][1] / r_D_e, position[p][2] / r_D_e
                
                for j in range(N_H):
                    for i in range(N_R):
                        r_ij = cp.sqrt(
                            (grid_R_gpu[i] - x) ** 2 + y ** 2 + (grid_H_gpu[j] - z) ** 2
                        )
                        potential_gpu[j][i] = cp.sum(cp.exp(-r_ij) / (r_ij + r_0))
                grid_R = cp.asnumpy(grid_R_gpu)
                potential = cp.asnumpy(potential_gpu) / float(N)

                alpha = np.zeros((5,N_H))
                for j in range(N_H):
                    A = np.vstack(
                        [
                            np.ones(len(grid_R)),
                            grid_R ** 2,
                            grid_R ** 4,
                            grid_R ** 6,
                            grid_R ** 8,
                        ]
                    ).T
                    potential_line = potential[j]
                    alpha[0][j], alpha[1][j], alpha[2][j], alpha[3][j], alpha[4][j] = -np.linalg.lstsq(A, potential_line, rcond=None)[0] * N
                return alpha
            if trapFileName == "":
                alphaTrap = computeAndFitBackgroundPotential(self.R, self.H, self.r_D_e, self.N, position)
            else:
                _alpha_0 = []
                _alpha_1 = []
                _alpha_2 = []
                _alpha_3 = []
                _alpha_4 = []
                _file = open(trapFileName, 'r')
                for _line in _file.readlines():
                    _alpha_0.append(float(_line.split("\t")[0]))
                    _alpha_1.append(float(_line.split("\t")[1]))
                    _alpha_2.append(float(_line.split("\t")[2]))
                    _alpha_3.append(float(_line.split("\t")[3]))
                    _alpha_4.append(float(_line.split("\t")[4]))
                alphaTrap = np.array([np.array(_alpha_0), np.array(_alpha_1), np.array(_alpha_2), np.array(_alpha_3), np.array(_alpha_4)])
            
            alpha_0 = mm.Continuous1DFunction(alphaTrap[0], -self.H/2.0, self.H/2.0)
            alpha_1 = mm.Continuous1DFunction(alphaTrap[1], -self.H/2.0, self.H/2.0)
            alpha_2 = mm.Continuous1DFunction(alphaTrap[2], -self.H/2.0, self.H/2.0)
            alpha_3 = mm.Continuous1DFunction(alphaTrap[3], -self.H/2.0, self.H/2.0)
            alpha_4 = mm.Continuous1DFunction(alphaTrap[4], -self.H/2.0, self.H/2.0)

            trapIonForceLine = "{}*(alpha_0(h) + alpha_1(h)*r^2+alpha_2(h)*r^4+alpha_3(h)*r^6+alpha_4(h)*r^8)".format(
                self.q_s
                * self.q_s
                / self.plasmaParametersInSIUnits.r_D_e
                * self.k
                * (1 * J).to("kJ/mol").value
            )

            trapIonForceLine += "; r = sqrt(x1^2+y1^2)/{}; h = z1/{}".format(
                self.r_D_e, 1
            )
            trapIonForce = mm.CustomCompoundBondForce(1,trapIonForceLine)
            trapIonForce.addTabulatedFunction("alpha_0", alpha_0)
            trapIonForce.addTabulatedFunction("alpha_1", alpha_1)
            trapIonForce.addTabulatedFunction("alpha_2", alpha_2)
            trapIonForce.addTabulatedFunction("alpha_3", alpha_3)
            trapIonForce.addTabulatedFunction("alpha_4", alpha_4)
            trapIonForce.setForceGroup(2)


        """ Define external electric field force """
        lengthConversion = (1 * nm).to("m").value
        if self.externalFieldOrder == 0:
            externalFieldForceLine = "-E_0*z*q_s; "
            externalFieldForceLine += "E_0={}; ".format(self.Phi_parameters['E_0'][0]*self.externalFieldConversionCoefficient)
        elif self.externalFieldOrder == 2:
            externalFieldForceLine = "(-z*(E_0 + alpha*z1 + beta*z1^2)*step(z1-z) - z1*(E_0 + alpha*z1 + beta*z1^2)*step(z-z1) - ((E_0*z + alpha*z^2/2 + beta*z^3/3)-(E_0*z1 + alpha*z1^2/2 + beta*z1^3/3))*step(z-z1)*step(z7-z)-((E_0*z7 + alpha*z7^2/2 + beta*z7^3/3)-(E_0*z1 + alpha*z1^2/2 + beta*z1^3/3))*step(z-z7)-(E_0 + alpha*z7 + beta*z7^2)*z*step(z-z7))*q_s; "
            #externalFieldForceLine = "(-(E_0 + alpha*z1 + beta*z1^2)-1/(1+exp(-(z-z1)/(0.02*r_D_e)))*(z-z1)^2*(3*alpha+2*beta*(2*z1+z))/6.0)*q_s; "
            externalFieldForceLine += "r_D_e={}; ".format(self.r_D_e)
            externalFieldForceLine += "z1={}; ".format(float((self.Phi_parameters['z1'][0] * m).to("nm").value))
            externalFieldForceLine += "z7={}; ".format(float((self.Phi_parameters['z7'][0] * m).to("nm").value))
            externalFieldForceLine += "E_0={}; ".format(self.Phi_parameters['E_0'][0]*self.externalFieldConversionCoefficient)
            externalFieldForceLine += "alpha={}; ".format(self.Phi_parameters['alpha'][0]*self.externalFieldConversionCoefficient * lengthConversion)
            externalFieldForceLine += "beta={}; ".format(self.Phi_parameters['beta'][0]*self.externalFieldConversionCoefficient * lengthConversion**2)
        externalFieldForceLine += "q_s={}".format(self.q_s)
        externalFieldForce = mm.CustomExternalForce(externalFieldForceLine)

        """ Aplly forces to system """
        for i in range(self.N):
            system.addParticle(self.m_s)

        for i in range(self.N):
            for p in range(self.numberOfDustParticles):
                dustParticleForces[p].addParticle(i, [])
            ionIonForce.addParticle([])
            if considerTrap:
                trapIonForce.addBond([i])
            externalFieldForce.addParticle(i, [])

        for p in range(self.numberOfDustParticles):
            system.addForce(dustParticleForces[p])
        system.addForce(ionIonForce)
        if considerTrap:
            system.addForce(trapIonForce)
        system.addForce(externalFieldForce)

        """ Define system integrator """

        def _vStringDefine(v_fl, v_T):

            v_max = v_fl + 4 * v_T
            v_min = max(v_fl - 4 * v_T, 0)

            if v_max < 0.4 * v_T:
                return 0, ""
            else:
                N_v = int((v_max - v_min) / (0.2 * v_T))
                v = np.linspace(v_min, v_max, N_v)
                xi = np.zeros(N_v)

                def f(v):
                    return (
                        v
                        / np.sqrt(2.0 * np.pi * v_T ** 2)
                        * np.exp(-((v - v_fl) ** 2) / (2.0 * v_T ** 2))
                    )

                def F(v):
                    return integrate.quad(f, v_min, v)[0]

                F_max = F(v_max)

                for i in range(N_v):
                    xi[i] = F(v[i]) / F_max


                v_string = "("

                for i in range(1, N_v):
                    if i == N_v - 1:
                        v_string += "({:.8f}+(_z(uniform)-{:.8f})*{:.8f})*step(-abs(_z(uniform)-{:.8f})+{:.8f})".format(
                            v[i - 1],
                            xi[i - 1],
                            (v[i] - v[i - 1]) / (xi[i] - xi[i - 1]),
                            0.5 * (xi[i] + xi[i - 1]),
                            0.5 * (xi[i] - xi[i - 1]),
                        )

                    else:
                        v_string += "({:.8f}+(_z(uniform)-{:.8f})*{:.8f})*step(-abs(_z(uniform)-{:.8f})+{:.8f}) + ".format(
                            v[i - 1],
                            xi[i - 1],
                            (v[i] - v[i - 1]) / (xi[i] - xi[i - 1]),
                            0.5 * (xi[i] + xi[i - 1]),
                            0.5 * (xi[i] - xi[i - 1]),
                        )

                v_string += ")"
                return v_string

        _x_axis = [mm.Vec3(1, 0, 0)] * self.N
        _y_axis = [mm.Vec3(0, 1, 0)] * self.N
        _z_axis = [mm.Vec3(0, 0, 1)] * self.N

        _vTopString = _vStringDefine(self.v_fl, self.v_T)

        integrator = mm.CustomIntegrator(self.d_t)
        integrator.addPerDofVariable("x_axis", 0.0)
        integrator.addPerDofVariable("y_axis", 0.0)
        integrator.addPerDofVariable("z_axis", 0.0)
        integrator.setPerDofVariableByName("x_axis", _x_axis)
        integrator.setPerDofVariableByName("y_axis", _y_axis)
        integrator.setPerDofVariableByName("z_axis", _z_axis)
        integrator.addGlobalVariable("H", self.H)
        integrator.addGlobalVariable("R", self.R)
        integrator.addGlobalVariable("v_T", self.v_T)
        integrator.addGlobalVariable("v_fl", self.v_fl)
        integrator.addGlobalVariable("pi", np.pi)
        integrator.addGlobalVariable("q_s", self.q_s)
        integrator.addGlobalVariable("k", self.k)
        integrator.addPerDofVariable("outside_domain_z", 0)
        integrator.addPerDofVariable("outside_domain_r", 0)
        integrator.addPerDofVariable("newx_r", 0)
        integrator.addPerDofVariable("newv_r", 0)
        integrator.addPerDofVariable("ions_to_sides", 0)
        integrator.addPerDofVariable("ions_to_correct", 0)
        integrator.addPerDofVariable("inside_dusts", 0)

        integrator.addPerDofVariable("n", 0)
        integrator.addPerDofVariable("tau", 0)

        integrator.addGlobalVariable("p_collision", self.d_t * self.w_c)
        
        integrator.addPerDofVariable("collision", 0)
        integrator.addComputePerDof("collision", "step(p_collision-uniform)") 
        integrator.addComputePerDof("v", "(1-collision)*v + collision*v_T*gaussian")
        
        integrator.addUpdateContextState()

        if considerTrap:
            integrator.addComputePerDof("v", "v+0.5*dt*f0/m")
            integrator.addComputePerDof("v", "v+0.5*dt*f1/m")
            integrator.addComputePerDof("v", "v+0.5*dt*f2/m")
            integrator.addComputePerDof("x", "x+dt*v")
            integrator.addComputePerDof("v", "v+0.5*dt*f0/m")
            integrator.addComputePerDof("v", "v+0.5*dt*f1/m")
            integrator.addComputePerDof("v", "v+0.5*dt*f2/m")

        else:
            integrator.addComputePerDof("v", "v+0.5*dt*f0/m")
            integrator.addComputePerDof("x", "x+dt*v")
            integrator.addComputePerDof("v", "v+0.5*dt*f0/m")


        for p in range(self.numberOfDustParticles):
            integrator.addPerDofVariable("inside_dust_p" + str(p), 0)
            integrator.addGlobalVariable("r_p" + str(p), self.dustParticles[p].r)
            integrator.addGlobalVariable("x_p" + str(p), self.dustParticles[p].x)
            integrator.addGlobalVariable("y_p" + str(p), self.dustParticles[p].y)
            integrator.addGlobalVariable("z_p" + str(p), self.dustParticles[p].z)
            integrator.addGlobalVariable("moment_f_x_p" + str(p), 0)
            integrator.addGlobalVariable("moment_f_y_p" + str(p), 0)
            integrator.addGlobalVariable("moment_f_z_p" + str(p), 0)
            integrator.addPerDofVariable("r_ps" + str(p), 0)
            integrator.addGlobalVariable("orbit_f_x_p" + str(p), 0)
            integrator.addGlobalVariable("orbit_f_y_p" + str(p), 0)
            integrator.addGlobalVariable("orbit_f_z_p" + str(p), 0)
            integrator.addGlobalVariable("delta_q_p" + str(p), 0)

        for p in range(self.numberOfDustParticles):
            integrator.addComputePerDof(
                "inside_dust_p" + str(p),
                "step(-sqrt((_x(x)-x_p{})^2+(_y(x)-y_p{})^2+(_z(x)-z_p{})^2)+r_p{})".format(
                    p, p, p, p
                ),
            )
        integrator.addComputePerDof(
            "outside_domain_r", "step(-R+sqrt(_x(x)^2+_y(x)^2))"
        )
        integrator.addComputePerDof("outside_domain_z", "step(-H/2+abs(_z(x)))")
        inside_dusts_string = ""
        for p in range(self.numberOfDustParticles):
            inside_dusts_string += "+inside_dust_p" + str(p)

        integrator.addComputePerDof("outside_domain_z", "outside_domain_z" + inside_dusts_string)

        for p in range(self.numberOfDustParticles):
            integrator.addComputeSum("delta_q_p" + str(p), "inside_dust_p" + str(p))
            integrator.addComputeSum(
                "moment_f_x_p" + str(p), "_x(v)*inside_dust_p" + str(p)
            )
            integrator.addComputeSum(
                "moment_f_y_p" + str(p), "_y(v)*inside_dust_p" + str(p)
            )
            integrator.addComputeSum(
                "moment_f_z_p" + str(p), "_z(v)*inside_dust_p" + str(p)
            )

        integrator.addComputePerDof("n", "vector(_x(x)/sqrt(_x(x)^2+_y(x)^2),_y(x)/sqrt(_x(x)^2+_y(x)^2),0)")
        integrator.addComputePerDof("tau", "vector(_y(x)/sqrt(_x(x)^2+_y(x)^2),-_x(x)/sqrt(_x(x)^2+_y(x)^2),0)")
        
        integrator.addComputePerDof(
            "v",
            "select(outside_domain_z,x_axis*v_T*_x(gaussian)+y_axis*v_T*_y(gaussian)+z_axis*"
            + _vTopString
            + ", v)",
        )
        integrator.addComputePerDof(
            "x",
            "select(outside_domain_z,x_axis*R*sqrt(_x(uniform))*cos(2*pi*_y(uniform))+y_axis*R*sqrt(_x(uniform))*sin(2*pi*_y(uniform))+z_axis*(-H/2), x)",
        )
        
        integrator.addComputePerDof("v","select(outside_domain_r, vector(-dot(v,n)*_x(n)+dot(v,tau)*_x(tau), -dot(v,n)*_y(n)+dot(v,tau)*_y(tau),_z(v)), v)")
        
        for p in range(self.numberOfDustParticles):
            integrator.addComputePerDof(
                "r_ps" + str(p),
                "sqrt((_x(x)-x_p"
                + str(p)
                + ")^2+(_y(x)-y_p"
                + str(p)
                + ")^2+(_z(x)-z_p"
                + str(p)
                + ")^2)",
            )

            integrator.addComputeSum(
                "orbit_f_x_p" + str(p),
                "exp(-r_ps"
                + str(p)
                + "/r_D_e)*(x_p"
                + str(p)
                + "-_x(x))/(r_ps"
                + str(p)
                + ")^3*r_D_e^2+exp(-r_ps"
                + str(p)
                + "/r_D_e)*(x_p"
                + str(p)
                + "-_x(x))/(r_ps"
                + str(p)
                + ")^2*r_D_e",
            )
            integrator.addComputeSum(
                "orbit_f_y_p" + str(p),
                "exp(-r_ps"
                + str(p)
                + "/r_D_e)*(y_p"
                + str(p)
                + "-_y(x))/(r_ps"
                + str(p)
                + ")^3*r_D_e^2+exp(-r_ps"
                + str(p)
                + "/r_D_e)*(y_p"
                + str(p)
                + "-_y(x))/(r_ps"
                + str(p)
                + ")^2*r_D_e",
            )

            integrator.addComputeSum(
                "orbit_f_z_p" + str(p),
                "exp(-r_ps"
                + str(p)
                + "/r_D_e)*(z_p"
                + str(p)
                + "-_z(x))/(r_ps"
                + str(p)
                + ")^3*r_D_e^2+exp(-r_ps"
                + str(p)
                + "/r_D_e)*(z_p"
                + str(p)
                + "-_z(x))/(r_ps"
                + str(p)
                + ")^2*r_D_e",
            )

        """ Create simulation context """
        platform = mm.Platform.getPlatformByName(platformName)
        properties = {'DeviceIndex': deviceIndex, 'Precision': 'single'}

        context = mm.Context(system, integrator, platform, properties)
        context.setPositions(position)
        context.setVelocities(velocity)

        """ Create output files """
        if self.outputParameters.xyzOutputFileName != "":
            xyzOutputFileName = self.outputParameters.xyzOutputFileName
            xyzOutputFile = open(xyzOutputFileName, "w")
            xyzOutputFile.close()
        if self.outputParameters.csvOutputFileName != "":
            csvOutputFileName = self.outputParameters.csvOutputFileName
        if self.outputParameters.restartFileName != "":
            restartFileName = self.outputParameters.restartFileName
        nOutput = self.outputParameters.nOutput
        nFileOutput = self.outputParameters.nFileOutput

        """ Start simulation """
        for i in range(self.n):

            integrator.step(1)

            """ Write log to console """
            if i % nOutput == 0:
                if i == 0:
                    print("#########################################")
                    print("###        Start simulation           ###")
                    print("#########################################")
                    print("\n")
                print("Step: {} out of {}".format(i, self.n))

            if i % nFileOutput == 0:
                """Get state"""
                state = context.getState(True)
                position_array = state.getPositions(True)
                x = np.transpose(position_array)[0]
                y = np.transpose(position_array)[1]
                z = np.transpose(position_array)[2]
                """ Write state to XYZ-file """
                if self.outputParameters.xyzOutputFileName != "":
                    xyzOutputFile = open(xyzOutputFileName, "a")
                    xyzOutputFile.write("{}\n".format(self.N))
                    xyzOutputFile.write("{}\n".format(int(i / nFileOutput)))
                    for j in range(self.N):
                        xyzOutputFile.write("Ar\t{}\t{}\t{}\n".format(x[j], y[j], z[j]))
                    xyzOutputFile.close()
                """ Write state to CSV-file """
                if self.outputParameters.csvOutputFileName != "":
                    csvOutputFile = open(
                        csvOutputFileName + "{}.csv".format(int(i / nFileOutput)), "w"
                    )
                    _writer = csv.writer(csvOutputFile)
                    _header = ["x", "y", "z", "density", "mass"]
                    _writer.writerow(_header)
                    for j in range(self.N):
                        _data = [x[j], y[j], z[j], self.plasmaParametersInSIUnits.n_inf*1e-27, self.N_s]
                        _writer.writerow(_data)
                    csvOutputFile.close()
            """ Write state to a restart file """
            if i == self.n - 1:
                if self.outputParameters.restartFileName != "":
                    restartFile = open(self.outputParameters.restartFileName, "w")
                    state = context.getState(True, True)
                    position_array = state.getPositions(True)
                    velocity_array = state.getVelocities(True)
                    x = np.transpose(position_array)[0]
                    y = np.transpose(position_array)[1]
                    z = np.transpose(position_array)[2]
                    v_x = np.transpose(velocity_array)[0]
                    v_y = np.transpose(velocity_array)[1]
                    v_z = np.transpose(velocity_array)[2]
                    for j in range(self.N):
                        restartFile.write(
                            "{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                x[j], y[j], z[j], v_x[j], v_y[j], v_z[j]
                            )
                        )
                    restartFile.close()
            """ Write state to a restart file """
            if i == self.n - 1:
                if self.outputParameters.trapFileName != "":
                    trapFile = open(self.outputParameters.trapFileName, "w")
                    for j in range(len(alphaTrap[1])):
                        trapFile.write("{}\t{}\t{}\t{}\t{}\n".format(alphaTrap[0][j], alphaTrap[1][j], alphaTrap[2][j], alphaTrap[3][j], alphaTrap[4][j]))
                    trapFile.close()

            """ Dust particles' charges calculation """
            for p in range(self.numberOfDustParticles):
                """Self-consistent charge calculation"""
                if self.dustParticles[p].chargeCalculationMethod == "consistent":
                    if i > 0:
                        # devide by 3 to avoid duplicating (Openmm specific)
                        _delta_q_p = (
                            integrator.getGlobalVariableByName("delta_q_p" + str(p))
                            / 3.0
                        )
                        _phi_p = (
                            self.k
                            * self.dustParticles[p].q[i - 1]
                            / float((self.dustParticles[p].r * nm).to("m").value)
                        )
                        _k_B = self.plasmaParametersInSIUnits.k_B
                        _n_inf = self.plasmaParametersInSIUnits.n_inf
                        _T_e = self.plasmaParametersInSIUnits.T_e
                        _m_e = self.plasmaParametersInSIUnits.m_e
                        _gamma_e = (
                            _n_inf
                            * (_T_e * _k_B / np.pi / _m_e / 2.0) ** (1.0 / 2.0)
                            * np.exp(-self.e * _phi_p / _T_e / _k_B)
                        )
                        _d_t = self.simulationParametersInSIUnits.d_t
                        _S = (
                            float((self.dustParticles[p].r * nm).to("m").value) ** 2
                            * 4
                            * np.pi
                        )
                        _delta_q = _delta_q_p * self.q_s + _gamma_e * _d_t * _S * self.e
                        self.dustParticles[p].q[i] = (
                            self.dustParticles[p].q[i - 1] + _delta_q
                        )
                        context.setParameter("q" + str(p), self.dustParticles[p].q[i])
                """ OML charge calculation """
                if self.dustParticles[p].chargeCalculationMethod == "oml":
                    if i > 0:
                        _phi_p = (
                            self.k
                            * self.dustParticles[p].q[i - 1]
                            / float((self.dustParticles[p].r * nm).to("m").value)
                        )
                        _k_B = self.plasmaParametersInSIUnits.k_B
                        _n_inf = self.plasmaParametersInSIUnits.n_inf
                        _T_e = self.plasmaParametersInSIUnits.T_e
                        _m_e = self.plasmaParametersInSIUnits.m_e
                        _u = (
                            self.plasmaParametersInSIUnits.v_fl
                            / self.plasmaParametersInSIUnits.v_T
                            / np.sqrt(2.0)
                        )
                        _T_n = self.plasmaParametersInSIUnits.T_n
                        _chi = self.e * _phi_p / _T_n / _k_B
                        _v_ti = self.plasmaParametersInSIUnits.v_T * np.sqrt(2.0)
                        _gamma_e = (
                            _n_inf
                            * (_T_e * _k_B / np.pi / _m_e / 2.0) ** (1.0 / 2.0)
                            * np.exp(-self.e * _phi_p / _T_e / _k_B)
                        )

                        _gamma_i = (
                            _n_inf
                            * _v_ti
                            * _u
                            / 4.0
                            * (
                                (1.0 + 1.0 / (2.0 * _u * 2) + _chi / _u ** 2)
                                * special.erf(_u)
                                + 1.0 / _u / np.sqrt(np.pi) * np.exp(-(_u ** 2))
                            )
                        )
                        _d_t = self.simulationParametersInSIUnits.d_t
                        _S = (
                            float((self.dustParticles[p].r * nm).to("m").value) ** 2
                            * 4
                            * np.pi
                        )
                        _delta_q = (
                            _gamma_i * _d_t * _S * abs(self.e)
                            + _gamma_e * _d_t * _S * self.e
                        )
                        self.dustParticles[p].q[i] = (
                            self.dustParticles[p].q[i - 1] + _delta_q
                        )
                        context.setParameter("q" + str(p), self.dustParticles[p].q[i])

            """ Calculate forces acting on dust particles """
            for p in range(self.numberOfDustParticles):
                """Dust-dust forces"""
                _forceFromDustsX = 0
                _forceFromDustsY = 0
                _forceFromDustsZ = 0
                for j in range(self.numberOfDustParticles):
                    if p != j:
                        _x_p = float((self.dustParticles[p].x * nm).to("m").value)
                        _y_p = float((self.dustParticles[p].y * nm).to("m").value)
                        _z_p = float((self.dustParticles[p].z * nm).to("m").value)
                        _x_j = float((self.dustParticles[j].x * nm).to("m").value)
                        _y_j = float((self.dustParticles[j].y * nm).to("m").value)
                        _z_j = float((self.dustParticles[j].z * nm).to("m").value)
                        _q_p = self.dustParticles[p].q[i]
                        _q_j = self.dustParticles[j].q[i]
                        _r_pj = np.sqrt(
                            (_x_p - _x_j) ** 2 + (_y_p - _y_j) ** 2 + (_z_p - _z_j) ** 2
                        )

                        _forceFromDustsX += (
                            self.k * _q_p * _q_j * (_x_p - _x_j) / _r_pj ** 3
                        )
                        _forceFromDustsY += (
                            self.k * _q_p * _q_j * (_y_p - _y_j) / _r_pj ** 3
                        )
                        _forceFromDustsZ += (
                            self.k * _q_p * _q_j * (_z_p - _z_j) / _r_pj ** 3
                        )
                self.dustParticles[p].forceDust[i][0] = _forceFromDustsX
                self.dustParticles[p].forceDust[i][1] = _forceFromDustsY
                self.dustParticles[p].forceDust[i][2] = _forceFromDustsZ

                """ Ion interaction forces """
                """ 1. In-domain contribution"""
                # devide by 3 to avoid duplicating (Openmm specific)
                _forceFromIonsOrbitsX = (
                    integrator.getGlobalVariableByName("orbit_f_x_p" + str(p)) / 3.0
                )
                _forceFromIonsOrbitsY = (
                    integrator.getGlobalVariableByName("orbit_f_y_p" + str(p)) / 3.0
                )
                _forceFromIonsOrbitsZ = (
                    integrator.getGlobalVariableByName("orbit_f_z_p" + str(p)) / 3.0
                )
                _forceFromIonsOrbitsX *= self.dustParticles[p].q[i] * self.k * self.q_s
                _forceFromIonsOrbitsY *= self.dustParticles[p].q[i] * self.k * self.q_s
                _forceFromIonsOrbitsZ *= self.dustParticles[p].q[i] * self.k * self.q_s
                _forceFromIonsOrbitsX /= self.plasmaParametersInSIUnits.r_D_e ** 2
                _forceFromIonsOrbitsY /= self.plasmaParametersInSIUnits.r_D_e ** 2
                _forceFromIonsOrbitsZ /= self.plasmaParametersInSIUnits.r_D_e ** 2

                """ 2. Out-domaint contribution """

                def outDomainIonForce(x, y, z, q, alpha, plasmaParameters, q_s):
                    r = np.sqrt(x ** 2 + y ** 2)
                    alpha_0 = UnivariateSpline(np.linspace(-self.H/self.r_D_e/2.0, self.H/self.r_D_e/2.0, len(alpha[0])),alpha[0], k=4, s=0)
                    alpha_1 = UnivariateSpline(np.linspace(-self.H/self.r_D_e/2.0, self.H/self.r_D_e/2.0, len(alpha[1])),alpha[1], k=4, s=0)
                    alpha_2 = UnivariateSpline(np.linspace(-self.H/self.r_D_e/2.0, self.H/self.r_D_e/2.0, len(alpha[2])),alpha[2], k=4, s=0)
                    alpha_3 = UnivariateSpline(np.linspace(-self.H/self.r_D_e/2.0, self.H/self.r_D_e/2.0, len(alpha[3])),alpha[3], k=4, s=0)
                    alpha_4 = UnivariateSpline(np.linspace(-self.H/self.r_D_e/2.0, self.H/self.r_D_e/2.0, len(alpha[4])),alpha[4], k=4, s=0)

                    alpha_0_z = alpha_0.derivative()
                    alpha_1_z = alpha_1.derivative()
                    alpha_2_z = alpha_2.derivative()
                    alpha_3_z = alpha_3.derivative()
                    alpha_4_z = alpha_4.derivative()

                    r_D_e = plasmaParameters.r_D_e
                    k = 1.0 / (plasmaParameters.eps_0 * 4.0 * np.pi)
                    
                    forceR = (-2 * alpha_1(z/ r_D_e) * r) / r_D_e ** 2
                    forceR += (-4 * alpha_2(z/ r_D_e) * r**3) / r_D_e**4
                    forceR += (-6 * alpha_3(z/ r_D_e) * r**5) / r_D_e**6
                    forceR += (-8 * alpha_4(z/ r_D_e) * r**7) / r_D_e**8

                    forceR *= k * q * q_s / r_D_e

                    forceH = (-alpha_0_z(z/ r_D_e)) / r_D_e
                    forceH += (-alpha_1_z(z/ r_D_e) * r**2) / r_D_e ** 2 / r_D_e
                    forceH += (-alpha_2_z(z/ r_D_e) * r**4) / r_D_e ** 4 / r_D_e
                    forceH += (-alpha_3_z(z/ r_D_e) * r**6) / r_D_e ** 6 / r_D_e
                    forceH += (-alpha_4_z(z/ r_D_e) * r**8) / r_D_e ** 8 / r_D_e

                    forceH *= (k * q * q_s / r_D_e)

                    if r == 0:
                        forceX = 0
                        forceY = 0
                    else:
                        forceX = forceR * x / r
                        forceY = forceR * y / r
                    forceZ = forceH
                    return forceX, forceY, forceZ

                _x_p = float((self.dustParticles[p].x * nm).to("m").value)
                _y_p = float((self.dustParticles[p].y * nm).to("m").value)
                _z_p = float((self.dustParticles[p].z * nm).to("m").value)
                _q_p = self.dustParticles[p].q[i]
                (
                    _forceFromIonsOutX,
                    _forceFromIonsOutY,
                    _forceFromIonsOutZ,
                ) = outDomainIonForce(
                    _x_p,
                    _y_p,
                    _z_p,
                    _q_p,
                    alphaTrap,
                    self.plasmaParametersInSIUnits,
                    self.q_s,
                )

                self.dustParticles[p].forceIonsOrbit[i][0] = (
                    _forceFromIonsOrbitsX + _forceFromIonsOutX
                )
                self.dustParticles[p].forceIonsOrbit[i][1] = (
                    _forceFromIonsOrbitsY + _forceFromIonsOutY
                )
                self.dustParticles[p].forceIonsOrbit[i][2] = (
                    _forceFromIonsOrbitsZ + _forceFromIonsOutZ
                )

                """ Ion collision forces """
                # devide by 3 to avoid duplicating (Openmm specific)
                _forceFromIonsCollisionsX = (
                    integrator.getGlobalVariableByName("moment_f_x_p" + str(p)) / 3.0
                )
                _forceFromIonsCollisionsY = (
                    integrator.getGlobalVariableByName("moment_f_y_p" + str(p)) / 3.0
                )
                _forceFromIonsCollisionsZ = (
                    integrator.getGlobalVariableByName("moment_f_z_p" + str(p)) / 3.0
                )
                _forceFromIonsCollisionsX *= (
                    float((1 * nm / ps).to("m/s").value)
                    * self.plasmaParametersInSIUnits.m_i
                    * self.N_s
                    / self.simulationParametersInSIUnits.d_t
                )
                _forceFromIonsCollisionsY *= (
                    float((1 * nm / ps).to("m/s").value)
                    * self.plasmaParametersInSIUnits.m_i
                    * self.N_s
                    / self.simulationParametersInSIUnits.d_t
                )
                _forceFromIonsCollisionsZ *= (
                    float((1 * nm / ps).to("m/s").value)
                    * self.plasmaParametersInSIUnits.m_i
                    * self.N_s
                    / self.simulationParametersInSIUnits.d_t
                )

                self.dustParticles[p].forceIonsCollect[i][0] = _forceFromIonsCollisionsX
                self.dustParticles[p].forceIonsCollect[i][1] = _forceFromIonsCollisionsY
                self.dustParticles[p].forceIonsCollect[i][2] = _forceFromIonsCollisionsZ

                """External electic field forces """
                _q_p = self.dustParticles[p].q[i]
                _z_p = float((self.dustParticles[p].z * nm).to("m").value)
                _forceExternalFieldX = 0
                _forceExternalFieldY = 0
                if self.externalFieldOrder == 0:
                    E_0 = self.Phi_parameters['E_0'][0]
                    _forceExternalFieldZ = _q_p * (E_0)
                elif self.externalFieldOrder == 2:
                    E_0 = self.Phi_parameters['E_0'][0]
                    alpha = self.Phi_parameters['alpha'][0]
                    beta = self.Phi_parameters['beta'][0]
                    _forceExternalFieldZ = _q_p * (E_0+ alpha* _z_p + beta* _z_p**2)

                self.dustParticles[p].forceExternalField[i][
                    0
                ] = _forceExternalFieldX
                self.dustParticles[p].forceExternalField[i][
                    1
                ] = _forceExternalFieldY
                self.dustParticles[p].forceExternalField[i][
                    2
                ] = _forceExternalFieldZ

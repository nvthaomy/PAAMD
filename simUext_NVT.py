from sys import stdout
import numpy as np
import mdtraj, simtk
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *

N_av = 6.022140857*10**23 /mole
kB = 1.380649*10**(-23)*joules/kelvin* N_av #joules/kelvin/mol

# Input Files
NPToutput = 'output298NPT.xml' 
fName = 'xp0.1_N12_f0.5_V157_LJPME_298K_NVT_Uext0'
prmtop = AmberPrmtopFile('AA12_f0.5_opc_gaff2_w0.13.parm7')
inpcrd = AmberInpcrdFile('AA12_f0.5_opc_w0.13.crd')

# System Configuration

nonbondedMethod = LJPME
nonbondedCutoff = 1.0*nanometers
tailCorrection = True
ewaldErrorTolerance = 0.0001
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001
box_vectors = np.diag([5.320503,5.320503,5.320503]) * nanometer
# Integration Options

dt = 0.002*picoseconds
temperature = 298.15*kelvin
friction = 1.0/(100.*dt)
pressure = 1.0*atmospheres
barostatInterval = 25

#External potential

Uext = 0. *kB*temperature
Nperiod = 1
axis  = 2
planeLoc = 0*nanometer
atomsInExtField = ['O']
resInExtField = ['HOH']

# Simulation Options

steps = 2e7
equilibrationSteps = 100000
#platform = Platform.getPlatformByName('CUDA')
#platformProperties = {'Precision': 'mixed'}
dcdReporter = mdtraj.reporters.DCDReporter('trajectory_{}.dcd'.format(fName), 5000)
dataReporter = StateDataReporter('log_{}.txt'.format(fName), 1000, totalSteps=steps, step=True, speed=True, progress=True, remainingTime=True, potentialEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, separator='\t')

# Prepare the Simulation

print('Building system...')
topology = prmtop.topology
positions = inpcrd.positions
system = prmtop.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,constraints=constraints, rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance)
integrator = LangevinIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)
#simulation = Simulation(topology, system, integrator, platform)
simulation = Simulation(topology, system, integrator)#, platform, platformProperties)
simulation.context.setPositions(positions)
if inpcrd.boxVectors is not None:
	simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

simulation.context.setPeriodicBoxVectors(*box_vectors)

#===========================
# Create external potential
#===========================
external={"U":Uext,"NPeriod":Nperiod,"axis":axis ,"planeLoc":planeLoc}
direction=['x','y','z']
ax = external["axis"]
#atomsInExtField = [elementMap[atomname]]
if external["U"] > 0.0 * kilojoules_per_mole:
        print('Creating sinusoidal external potential in the {} direction'.format(direction[axis]))
        energy_function = 'U*sin(2*pi*NPeriod*({axis}-r0)/{L})'.format(axis=direction[ax],L=box_vectors[ax][ax])
        fExt = openmm.CustomExternalForce(energy_function)
        fExt.addGlobalParameter("U", external["U"])
        fExt.addGlobalParameter("NPeriod", external["NPeriod"])
        fExt.addGlobalParameter("pi",np.pi)
        fExt.addGlobalParameter("r0",external["planeLoc"])
        fExt.addGlobalParameter("L",box_vectors[ax][ax])

        for ia,atom in enumerate(topology.atoms()):
             if atom.name in atomsInExtField and atom.residue.name in resInExtField:
                fExt.addParticle( ia,[] )
        system.addForce(fExt)
        print('Number of atoms in fExt %i' %fExt.getNumParticles())

forces = system.getForces()
for force in forces:
    if isinstance(force,simtk.openmm.openmm.NonbondedForce):
        nonbondedforce = force
nonbondedforce.setUseDispersionCorrection(tailCorrection)
nonbondedforce.updateParametersInContext(simulation.context)
forces = system.getForces()
for force in forces:
    if isinstance(force,simtk.openmm.openmm.NonbondedForce):
        nonbondedforce = force
print('getUseDispersionCorrection')
print(nonbondedforce.getUseDispersionCorrection())
#Restart and Check point

#to load state: simulation.loadState('output.xml')
simulation.loadState(NPToutput)
simulation.reporters.append(CheckpointReporter('checkpnt_{}.chk'.format(fName), 5000))

# Minimize and Equilibrate

print('Performing energy minimization...')
simulation.minimizeEnergy()
print('Equilibrating...')
simulation.context.setVelocitiesToTemperature(temperature)
simulation.step(equilibrationSteps)
simulation.saveState('output_warmup_{}.xml'.format(fName))
# Simulate

print('Simulating...')
simulation.reporters.append(dcdReporter)
simulation.reporters.append(dataReporter)
simulation.currentStep = 0
simulation.step(steps)
simulation.saveState('output_{}.xml'.format(fName))

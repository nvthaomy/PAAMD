import numpy as np
from sys import stdout
import mdtraj
import simtk
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *

N_av = 6.022140857*10**23 /mole
kB = 1.380649*10**(-23)*joules/kelvin* N_av #joules/kelvin/mol

#output name
fName = 'xp0.1_N12_f0_V328_LJPME_298K_UgaussPP5'

# Input Files

prmtop = AmberPrmtopFile('AA12_f0_opc_gaff2_w0.13.parm7')
inpcrd = AmberInpcrdFile('AA12_f0_opc_w0.13.crd')
DOP = 12
PolyResName = ['AP','AHP','ATP','AD','AHD','ATD']
# System Configuration

nonbondedMethod = LJPME
nonbondedCutoff = 0.9*nanometers
tailCorrection = False
ewaldErrorTolerance = 0.0001
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001
box_vectors = np.diag([5.4,5.4,11.216]) * nanometer

# Integration Options

dt = 0.002*picoseconds #0.002
temperature = 298.15*kelvin
friction = 1.0/(100.*dt)
pressure = 1.0*atmospheres
barostatInterval = 25


#External potential: applying sinusoidal external potential on polymer centroid, set Uext = 0 to disable 
mapping = 1 #apply the external potential on the centroid of this many polymer monomers
Uext = 0. *kB*temperature #Uext on each centroid
Nperiod = 1
axis  = 2
planeLoc = 0*nanometer
#atomsInExtField = ['CB']
resInExtField = ['AP','AHP','ATP','AD','AHD','ATD']

#additional gaussian attraction/repulsion between polymer COM and solvent COM
mapping1 = 1 
B_PW = 0. *kB*temperature
aevs = [0.31,0.45] #excluded volume a in nanometer of 2 species in the gaussian 
Kappa_PW = 1/(4* (np.mean(aevs) * nanometer)**2)# kappa = 1/(4 a_ev**2) 

#additional gaussian attraction/repulsion between polymer COM and solvent COM
mapping2 = 1
B_PP = 5. *kB*temperature
aevs = [0.45,0.45] #excluded volume a in nanometer of 2 species in the gaussian 
Kappa_PP = 1/(4* (np.mean(aevs) * nanometer)**2)# kappa = 1/(4 a_ev**2) 

# Simulation Options

steps = 2e6 #3e7
equilibrationSteps = 500000
#platform = Platform.getPlatformByName('CUDA')
#platformProperties = {'Precision': 'mixed'}
dcdReporter = mdtraj.reporters.DCDReporter('trajectory_{}.dcd'.format(fName), 1000)
dataReporter = StateDataReporter('log_{}.txt'.format(fName), 1000, totalSteps=steps, step=True, speed=True, progress=True, remainingTime=True, potentialEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, separator='\t')

# Prepare the Simulation

print('Building system...')
topology = prmtop.topology
positions = inpcrd.positions
system = prmtop.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,constraints=constraints, rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance)

#=============================
# Create external potential
#=============================
external={"U":Uext,"NPeriod":Nperiod,"axis":axis ,"planeLoc":planeLoc}
direction=['x','y','z']
ax = external["axis"]
#atomsInExtField = [elementMap[atomname]]
if external["U"] > 0.0 * kilojoules_per_mole:
        energy_function = 'U*sin(2*pi*NPeriod*({axis}1-r0)/L)'.format(axis=direction[ax])
        fExt = openmm.CustomCentroidBondForce(1,energy_function)
        fExt.addGlobalParameter("U", external["U"])
        fExt.addGlobalParameter("NPeriod", external["NPeriod"])
        fExt.addGlobalParameter("pi",np.pi)
        fExt.addGlobalParameter("r0",external["planeLoc"])
        fExt.addGlobalParameter("L",box_vectors[ax][ax])
        
        #get number of polymer chain
        NP = 0
        for ir, res in enumerate(topology.residues()):
            if res.name in resInExtField:
                NP += 1
        NP /= DOP
        aIds = []
        for i in range(int(NP*DOP/mapping)): #looping through CG beads
            aId = []
            resIDs = range(i*mapping, (i+1)*mapping) #looping through AA monomers in this CG bead
            for res in topology.residues():
                 if res.index in resIDs:
                     for atom in res.atoms():
                         aId.append(atom.index)
            aIds.append(aId)
            fExt.addGroup(aId) #by default, using particle masses as weights 
            fExt.addBond([i], [])
#        print(aIds)
#        np.savetxt('uext_AtomId.dat',aIds,header='Atom ids in Uext')
        system.addForce(fExt)

#=========================================================================
# Create Gaussian attraction/repulsion between polymer COM and solvent COM
#=========================================================================
gaussPW={"B_PW": B_PW, "Kappa_PW": Kappa_PW}
if np.abs(gaussPW["B_PW"]) > 0.0 * kilojoules_per_mole:
        energy_function1 = 'B_PW*exp(-Kappa_PW*distance(g1,g2)^2)'
        fGauss = openmm.CustomCentroidBondForce(2,energy_function1)
        fGauss.addGlobalParameter("B_PW", gaussPW["B_PW"])
        fGauss.addGlobalParameter("Kappa_PW", gaussPW["Kappa_PW"])

        #get number of polymer chain
        NP = 0
        for ir, res in enumerate(topology.residues()):
            if res.name in PolyResName:
                NP += 1
        NP /= DOP
        NBeads = int(NP*DOP/mapping1)
        PolyIds = []
        #adding polymer beads
        for i in range(NBeads): #looping through CG beads
            PolyId = []
            resIDs = range(i*mapping1, (i+1)*mapping1) #looping through AA monomers in this CG bead
            for res in topology.residues():
                 if res.index in resIDs:
                     for  atom in res.atoms():
                         PolyId.append(atom.index)
            PolyIds.append(PolyId)
            fGauss.addGroup(PolyId) #by default, using particle masses as weights 
#        np.savetxt('gaussPW_PolyId.dat',PolyIds,fmt='%i',header='Polymer ids in gaussPW')
    
        #adding all solvent 
        NW = 0
        WatIds = []
        print('adding water atoms')
        for ir, res in enumerate(topology.residues()):
            if res.name in ['WAT', 'HOH']:
                NW += 1
                WatId = []
                for atom in res.atoms():
                    WatId.append(atom.index)
                WatIds.append(WatId)
                fGauss.addGroup(WatId)
#        np.savetxt('gaussPW_WatId.dat',WatIds,fmt='%i',header='Water ids in gaussPW')

        #adding bonds to the potential
        for i in range(0,NBeads):
            for j in range(NBeads,NBeads+NW):
                fGauss.addBond([i,j],[]) #by default, using particle masses as weights
        print('{} Gaussian interactions were added for {} polymer beads and {} water molecules'.format(fGauss.getNumBonds(),NBeads,NW))
        print(gaussPW)
        fGauss.setUsesPeriodicBoundaryConditions(True)        
        system.addForce(fGauss)

#=========================================================================
# Create Gaussian attraction/repulsion between COM of polymer ends
#=========================================================================
gaussPP={"B_PP": B_PP, "Kappa_PP": Kappa_PP}
if np.abs(gaussPP["B_PP"]) > 0.0 * kilojoules_per_mole:
        energy_function2 = 'B_PP*exp(-1* Kappa_PP *distance(g1,g2)^2)'
        fGauss = openmm.CustomCentroidBondForce(2,energy_function2)
        fGauss.addGlobalParameter("B_PP", gaussPP["B_PP"])
        fGauss.addGlobalParameter("Kappa_PP", gaussPP["Kappa_PP"])

        #get number of polymer chain
        NP = 0
        for ir, res in enumerate(topology.residues()):
            if res.name in resInExtField:
                NP += 1
        NP /= DOP
        NBeads = int(NP*DOP/mapping2)
        CGDOP = int(float(NBeads)/float(NP))
        
        #adding polymer beads
        bondgroups = []
        for i in range(NBeads): #looping through CG beads
                PolyId = []
#            if i%CGDOP == 0 or (i+1)%CGDOP == 0: #only add end monomers to this force
                resIDs = range(i*mapping2, (i+1)*mapping2) #looping through AA monomers in this CG bead
                for res in topology.residues():
                     if res.index in resIDs:
                         for atom in res.atoms():
                             PolyId.append(atom.index)
                fGauss.addGroup(PolyId) #by default, using particle masses as weights 

        #adding bonds to the potential
        for i in range(fGauss.getNumGroups()):
            for j in range(i+1, fGauss.getNumGroups()):
                fGauss.addBond([i,j],[]) #by default, using particle masses as weights
                bondgroups.append([i,j])
        np.savetxt('bondgroups.dat',bondgroups, fmt='%i%')
        print('{} Gaussian interactions were added for {} polymer beads'.format(fGauss.getNumBonds(),fGauss.getNumGroups()))
        print(gaussPP)
        fGauss.setUsesPeriodicBoundaryConditions(True)
        system.addForce(fGauss)


integrator = LangevinIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)
simulation = Simulation(topology, system, integrator) #, platform, platformProperties)
simulation.context.setPositions(positions)
if inpcrd.boxVectors is not None:
	simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

simulation.context.setPeriodicBoxVectors(*box_vectors)

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
print('getNonbondedMethod')
print(nonbondedforce.getNonbondedMethod())
#print('getLJPMEParametersInContext')
#print(nonbondedforce.getLJPMEParametersInContext(simulation.context))

#Restart and Check point
simulation.loadState('output298NVT_init.xml')
#simulation.loadCheckpoint('checkpnt_2Mnacl_opc_Uext1_298K.chk')
simulation.reporters.append(CheckpointReporter('checkpnt_{}.chk'.format(fName), 5000))

# Minimize and Equilibrate
state = simulation.context.getState(getPositions=True,getEnergy=True)
simulation.saveState('output_init_{}.xml'.format(fName))
print("PE {}".format(state.getPotentialEnergy()))

print ("Periodic box vector: {}".format(state.getPeriodicBoxVectors()))
#simulation.saveState('output_warmup_{}.xml'.format(fName))

print('Performing energy minimization...')
simulation.minimizeEnergy()
state = simulation.context.getState(getPositions=True)
PDBFile.writeModel(simulation.topology, state.getPositions(), open('traj_minimized_{}.pdb'.format(fName),'w'))

print('Equilibrating...')
#simulation.context.setVelocitiesToTemperature(temperature)
#simulation.step(equilibrationSteps)
#simulation.saveState('output_warmup_{}.xml'.format(fName))
# Simulate

print('Simulating...')
simulation.reporters.append(dcdReporter)
simulation.reporters.append(dataReporter)
simulation.currentStep = 0
simulation.step(steps)
simulation.saveState('output_{}.xml'.format(fName))

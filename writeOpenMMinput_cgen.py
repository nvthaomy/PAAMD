#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:50:27 2019

@author: nvthaomy
"""
import glob
import os, shutil
import time
import genPackmol as packmol
import fixgmxtop
def writeJobfile(temp,charge,w,N):
    '''write slurm job file for Pod cluster'''
    name = 'job{}'.format(int(temp))
    charge = ''.join(str(charge).split('.'))
    w = ''.join(str(w).split('.'))
    with open(name,'w') as job:
        job.write('#!/bin/bash')
        job.write('\n#SBATCH -N 1 --partition=gpu --ntasks-per-node=1')   
        job.write('\n#SBATCH --gres=gpu:1')
        job.write('\n#SBATCH -J A{}f{}w{}{}K\n'.format(N,charge,w,int(temp)))
        job.write('\ncd $SLURM_SUBMIT_DIR\n')        
        job.write('\n/bin/hostname')
        job.write('\nsrun --gres=gpu:1 /usr/bin/nvidia-smi')
        job.write('\nexport PATH="/home/mnguyen/anaconda3/bin:$PATH"')
        job.write('\npython sim{}.py'.format(int(temp)))
    return name
def writeOpenMMinput(temp,top,gro):
    name = 'sim{}.py'.format(int(temp))
    with open(name,'w') as sim:
        sim.write('from sys import stdout')
        sim.write('\nfrom simtk.openmm import *')
        sim.write('\nfrom simtk.openmm.app import *')
        sim.write('\nimport parmed')
        sim.write('\nfrom simtk.unit import *\n')
        sim.write('\n# Input Files\n')
        sim.write('\nstrucFile = "{}"'.format(gro))
        sim.write('\ntopFile = "{}"'.format(top))
        sim.write('\ntop = parmed.load_file(topFile, xyz = strucFile)\n')
        sim.write('\n# System Configuration\n')
        sim.write('\nnonbondedMethod = PME')
        sim.write('\nnonbondedCutoff = 1.0*nanometers')
        sim.write('\newaldErrorTolerance = 0.0001')
        sim.write('\nconstraints = HBonds')
        sim.write('\nrigidWater = True')
        sim.write('\nconstraintTolerance = 0.000001\n')
        sim.write('\n# Integration Options\n')
        sim.write('\ndt = 0.002*picoseconds')
        sim.write('\ntemperature = {}*kelvin'.format(temp))
        sim.write('\nfriction = 1.0/(100*dt)')
        sim.write('\npressure = 1.0*atmospheres')
        sim.write('\nbarostatInterval = 25\n')
        sim.write('\n# Simulation Options\n')                    
        sim.write('\nsteps = 1e7')
        sim.write('\nequilibrationSteps = 100000')
        sim.write('\nplatform = Platform.getPlatformByName(\'CUDA\')')
        sim.write('\nplatformProperties = {\'Precision\': \'mixed\'}')
        sim.write('\ndcdReporter = DCDReporter(\'trajectory{}.dcd\', 5000)'.format(int(temp)))
        sim.write('\ndataReporter = StateDataReporter(\'log{}.txt\', 1000, totalSteps=steps, step=True, speed=True, progress=True, remainingTime=True, potentialEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, separator=\'\\t\')\n'.format(int(temp)))
        sim.write('\n# Prepare the Simulation\n')
        sim.write('\nprint(\'Building system...\')')
        sim.write('\ntopology = top.topology')
        sim.write('\npositions = top.positions')
        sim.write('\nsystem = top.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,constraints=constraints, rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance)')
        sim.write('\nsystem.addForce(MonteCarloBarostat(pressure, temperature, barostatInterval))')
        sim.write('\nintegrator = LangevinIntegrator(temperature, friction, dt)')
        sim.write('\nintegrator.setConstraintTolerance(constraintTolerance)')
        sim.write('\n#simulation = Simulation(topology, system, integrator, platform)')
        sim.write('\nsimulation = Simulation(topology, system, integrator, platform, platformProperties)')
        sim.write('\nsimulation.context.setPositions(positions)\n')
        #sim.write('\nif inpcrd.boxVectors is not None:')
        #sim.write('\n\tsimulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)\n')
        sim.write('\n#Restart and Check point')
        sim.write('\nsimulation.saveState(\'output{}.xml\')\n'.format(int(temp)))
        sim.write('\n#to load state: simulation.loadState(\'output.xml\')')
        sim.write('\nsimulation.reporters.append(CheckpointReporter(\'checkpnt{}.chk\', 5000))\n'.format(int(temp)))
        sim.write('\n# Minimize and Equilibrate\n')
        sim.write('\nprint(\'Performing energy minimization...\')')
        sim.write('\nsimulation.minimizeEnergy()')
        sim.write('\nprint(\'Equilibrating...\')')
        sim.write('\nsimulation.context.setVelocitiesToTemperature(temperature)')
        sim.write('\nsimulation.step(equilibrationSteps)\n')
        sim.write('\n# Simulate\n')
        sim.write('\nprint(\'Simulating...\')')
        sim.write('\nsimulation.reporters.append(dcdReporter)')
        sim.write('\nsimulation.reporters.append(dataReporter)')
        sim.write('\nsimulation.currentStep = 0')
        sim.write('\nsimulation.step(steps)')
        sim.write('\nsimulation.saveState(\'output{}.xml\')\n'.format(int(temp)))
    return name

def main(f,N,np,nw,w,watermodel,singleChainPdb,T):
    '''need to be execute in folder with charmm36.ff and ions.mdp'''
    if watermodel == 'opc':
        gmx_water = 'tip4p.gro'
        gmxwat = 'tip4p'
    elif watermodel == 'tip3p' or watermodel == 'spc' or watermodel == 'spce':
        gmx_water = 'spc216.gro'
        gmxwat = 'tip3p'
    mixturePdb=packmol.packmol_noSolvent(f,N,np,singleChainPdb)
    for pdb in singleChainPdb: #checking if single chain pdb files and packmol.job exist
        print pdb
        while not os.path.exists(pdb) or not os.path.exists('packmol.job'):  
            time.sleep(1)
    print('\nSubmitting packmol job ...')
    os.system('qsub packmol.job') 
    homeDir = os.getcwd()
    for pdb in mixturePdb:
        while not os.path.exists(pdb):  
            time.sleep(1) 
    print('\nDone packing molecules, setting up folders and generate topology files')
    for charge in f:
        for i,wt_fraction in enumerate(w):
            folder = 'f{}/w{}'.format(charge,wt_fraction)
            if not os.path.exists(folder): 
                    os.makedirs(folder)
            pdb = 'AA{}_f{}_np{}.pdb'.format(N,charge,int(np[i]))
            shutil.move(pdb,folder+'/'+pdb)
            if not os.path.exists(folder+'/charmm36.ff'):
                shutil.copytree('charmm36.ff',folder+'/charmm36.ff')
            shutil.copy('ions.mdp',folder+'/ions.mdp')
    os.system('\nmodule load gromacs/2016.1')
    time.sleep(10)
    print('Gromacs module is loaded')
    for charge in f:
        for i,wt_fraction in enumerate(w):
            folder = 'f{}/w{}'.format(charge,wt_fraction)
            os.chdir(folder)
            pdb = glob.glob('AA*f*np*pdb')[0]
            gro = 'AA{}_f{}_{}_w{}.gro'.format(N,charge,watermodel,wt_fraction)
            top = 'AA{}_f{}_{}_cgen_w{}.top'.format(N,charge,watermodel,wt_fraction)
            folder = 'f{}/w{}'.format(charge,wt_fraction)   
            #make top file
            os.system('\ngmx pdb2gmx -f {}  -ff charmm36 -p top.top -o {} -water {}'.format(pdb,gro,gmxwat))
            while not os.path.exists(gro) or not os.path.exists('top.top'):  
                time.sleep(1)
            #define box          
            os.system('\ngmx editconf -f {} -o {} -c -d 0.6 -bt cubic'.format(gro,gro))
            time.sleep(10)
            os.system('\ngmx solvate -cp {} -cs {} -o {} -p top.top -maxsol {}'.format(gro,gmx_water,gro,int(nw[i])))
            time.sleep(10)
            #fix function number for dihedral and angle descriptions in top file
            print 'np'
            print int(np[i])
            print 'wt_fraction'
            print wt_fraction
            if int(np[i]) == 1:
                print ('\nEditing topology file')
                outfile = 'top.top'
                infile = 'top_old.top'
                os.rename(outfile, infile)
                fixgmxtop.main(infile,outfile)
            else:
                print ('\nEditing topology files')
                chainItp = glob.glob('top*Other_chain*itp')
                print chainItp
                for itp in chainItp:
                    outfile = itp
                    infile = itp[:itp.index('.itp')]+'_old'+itp[itp.index('.itp'):]
                    os.rename(outfile, infile)
                    fixgmxtop.main(infile,outfile)
            
            print ('\nGenerate ions.tpr')
            os.system('\ngmx grompp -f ions.mdp -c {} -p top.top -o ions.tpr'.format(gro))
            while not os.path.exists('ions.tpr'):  
                time.sleep(1) 
            time.sleep(10)
            os.system('\ngmx genion -s ions.tpr -o {} -p top.top -pname NA  -np {}'.format(gro,int(charge*N*(np[i]))))
            time.sleep(10)
            with open('top.top','r') as oldTop:
                print '\nIncluding correct water.itp file in topology'
                watersection = False
                with open(top,'w') as newTop:
                    line = oldTop.readline()
                    while len(line):
                        if watersection:
                            newTop.write('\n#include "./charmm36.ff/{}.itp"'.format(watermodel))
                            watersection = False
                        else:
                            newTop.write(line)
                        if 'Include water topology' in line:
                            watersection = True
                        line = oldTop.readline()
            for temp in T:
                writeOpenMMinput(temp,top,gro)
                writeJobfile(temp,charge,wt_fraction,N)
            os.chdir(homeDir)

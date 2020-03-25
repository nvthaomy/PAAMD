#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:46:57 2019

@author: nvthaomy
"""
"""assumptions: all atoms of the same residues are listed in a sequential order"""
#flow of the scrip:
#input: .ac file : have bond connectivity, charges, 
# might need to read bond connectivity if doing QM calculation since output does not retain info about residue names
#1) readAC to get info like: total number of atom, posInd0, bondInd0, atomidBounds, atomTypesGen, bonds
#2) getAtomType
#3) getAvgCharge
#1) trim pdb file so that for each residue, write out a new pdf file containing only that residue
#   fix CHARGE
#   fix bond info
#   read bond connectivity
#   generate multiple pdb files
#for each residue:
    #2) run prepgen
    #   write .mc file: info of mainchain
    #   dont need to ommit any atoms since already do that in 1)
    #3) read charge according to atom type
    #4) store charge for each atom type in a matrix
import numpy as np
import os, time

def readAC(ac, nAtomList, Ctypes, Htypes, Otypes):
    """get info from ac file
        write out new file if there is more than one occurence of an atom name"""
    
    f = open(ac,'r')
    lines = f.readlines()
    f.close()
    
    nRes = len(nAtomList)
    #finding bound of atom id for each residue
    atomidBounds = [] #list of lower and upper bounds of atom id for all residues
    lowBound = 1
    for nAtom in nAtomList:
        atomidBounds.append([lowBound, lowBound + nAtom - 1])
        lowBound += nAtom 
    #geting info about number of residues and store all lines corresponding to a residue in a separate list
    nEleList = np.zeros([nRes,3], dtype = int) #list of list of number of each element for all residue in the order carbon, hydrogen, oxygen
    resLines = [] # list of lists of line containing info of positions and bond for each residue
    totChargeList = [0] * nRes #list of total charge per residue
    posInd0 = False
    bondInd0 = False
    nAtomTot = 0
    atomTypesGen = [] #default atom type in ac file
    charges = []
    atomNames = []
    for i, line in enumerate(lines):
        #find the first line of the position data and bond
        if  'ATOM' in line.split() or 'HETATM' in line.split():  
            nAtomTot += 1
            if not posInd0:
                posInd0 = i
            cols = line.split()          
            atomID = int(cols[1])
            atomType = cols[-1]
            charge = float(cols[-2]) 
            atomNames.append(cols[2])
            atomTypesGen.append(cols[-1])
            charges.append(charge)
            for j, bounds in enumerate(atomidBounds):
                if atomID >= bounds[0] and atomID <= bounds[-1]:
                    resID = j        
            #calculate charge per residue
            totChargeList[resID] += charge
            #counting number of each element in a residue
            if  atomType in Ctypes:             
                nEleList[resID][0] += 1
            elif  atomType in Htypes:
                nEleList[resID][1] += 1
            elif  atomType in Otypes:
                nEleList[resID][2] += 1 
        elif line.startswith('BOND'):
            cols = line.split()
            if not bondInd0:
                bondInd0 = i
                bonds = np.zeros([nAtomTot, nAtomTot], dtype = int)
            atomID1 = int(cols[2])
            atomID2 = int(cols[3])
            #filling up both lower and upper triangle of the bonds matrix
            bonds[atomID1-1][atomID2-1] = 1
            bonds[atomID2-1][atomID1-1] = 1    
                
    #if there is more than one occurence of an atom name, rewrite the .ac file with new atom names
    uniqueNames, counts = np.unique(atomNames, return_counts=True)
    if any(count > 1 for count in counts):
        print('There are more than one occurence of an atom name, writing out new file with unique atom names')
        Ccount = 1
        Hcount = 1
        Ocount = 1
        for i, at in enumerate(atomTypesGen):
            if at in Ctypes:
                atomNames[i] = 'C%i' %Ccount
                Ccount += 1
            elif at in Htypes:
                atomNames[i] = 'H%i' %Hcount
                Hcount += 1
            elif at in Otypes:
                atomNames[i] = 'O%i' %Ocount
                Ocount += 1
            else:
                raise Exception('Do not recognize %s atom type for atom id %i' %(at,i))
        print('New atom names \n{}'.format(atomNames))
        s = ''
        for i, line in enumerate(lines):
            if  'ATOM' in line.split() or 'HETATM' in line.split():
                cols = line.split()
                atomID = int(cols[1])
                aname = cols[2]
                space0 = (3 - len(aname)) * ' '
                space1 = (3 -len(atomNames[atomID-1])) * ' '
                line = line.replace(aname + space0, atomNames[atomID-1] + space1) #to conserve the space
                #atomID = int(cols[1])
                #cols[2] = atomNames[atomID-1]
                #line = '\t'.join(cols)
                #line += '\n'
            elif 'BOND' in line.split():
                cols = line.split()
                atomID1 = int(cols[2])
                atomID2 = int(cols[3])
                aname1 = cols[-2]
                aname2 = cols[-1]
                space01 = (3 - len(aname1)) * ' '
                space02 = (3 - len(aname2)) * ' '
                space11 = (3 -len(atomNames[atomID1-1])) * ' '
                space12 = (3 -len(atomNames[atomID2-1])) * ' '
                line = line.replace(space01+aname1, space11+atomNames[atomID1-1])
                line = line.replace(space02+aname2, space12+atomNames[atomID2-1])               
            s += line
        acOut = ac.split('.ac')[0] + '_newAtomTypes.ac'
        f = open(acOut,'w')
        f.write(s)
        ac = acOut


    #-------  writing ac file of each residue----#          
    #parsing position data    
    #currentLine = posInd0
    #for nAtom in nAtomList:
    #    resLines.append(lines[currentLine : currentLine + nAtom])
    #    currentLine += nAtom
  
    #parsing bond data, only add bonds connecting atoms of the same residue to list
    #bondLinesDict = {} #dictionary of lists of line containing info bond for each residue  
    #for i in range(nRes):
    #    bondLinesDict.update({'line{}'.format(i): []})
    #for i, line in enumerate(lines[bondInd0:]):
    #    cols = line.split()
    #    atomIDs = [int(cols[2]), int(cols[3])]
    #    for j, bounds in enumerate(atomidBounds):
    #        result = all(id >= bounds[0] and id <= bounds[-1] for id in atomIDs)
    #        if result:
    #            bondLinesDict['line{}'.format(j)].append(line) 
    #            break
    
    #for i, line in enumerate(resLines):
    #    line.extend(bondLinesDict['line{}'.format(i)])
        
    #for i, res in enumerate(range(nRes)):
    #    s = 'CHARGE      %5.6f\n'%totChargeList[i]
    #    s += 'Formula: H%i C%i O%i \n' %(nEleList[i][1], nEleList[i][0], nEleList[i][2])
    #    s += ''.join(resLines[i])
    #    acOut = ac.split('.ac')[0] + '_res{}.ac'.format(i+1)
    #    fOut = open(acOut,'w')
    #    fOut.write(s)
    return(posInd0, bondInd0, nAtomTot, atomidBounds, charges, atomTypesGen, bonds, atomNames, ac)

def getAtomType(bonds, nAtomTot, atomTypeDict, atomTypesGen):
    """assign atom types specific to PAA monomer to each atom"""

    atomTypes = [0.] * nAtomTot #PAA atom type
    
    #get atom type of PAA
    #assign atom type CB:
    for i, at in enumerate(atomTypesGen):
        if at == atomTypeDict['CB']:
            atomTypes[i] = 'CB'
        elif at == atomTypeDict['O']:
            atomTypes[i] = 'O'
        elif at == atomTypeDict['OH']:
            atomTypes[i] = 'OH'
        elif at == atomTypeDict['HO']:
            atomTypes[i] = 'HO'

    #assign atom type CA:
    for i, at in enumerate(atomTypesGen):
        #find atom that has not been assigned the type
        if not isinstance(atomTypes[i],str):
            bondedId = np.where(bonds[i] > 0.)[0].tolist() #index of atoms that the current bonded to
            bondedAtoms  = []
            for Id in bondedId: 
                bondedAtoms.append(atomTypes[Id]) #PAA atom types of atoms that the current atom bonded to 
            if at == atomTypeDict['CA'] and 'CB' in bondedAtoms:
                atomTypes[i] = 'CA'       
    #assign atom type CC:
    for i, at in enumerate(atomTypesGen):
        #find atom that has not been assigned the type
        if not isinstance(atomTypes[i],str):
            bondedId = np.where(bonds[i] > 0.)[0].tolist() #index of atoms that the current bonded to
            bondedAtoms  = []
            for Id in bondedId: 
                bondedAtoms.append(atomTypes[Id])
            if at == atomTypeDict['CC'] and 'CA' in bondedAtoms:
                atomTypes[i] = 'CC'     
    #assign atom type HA and HC:
    for i, at in enumerate(atomTypesGen):
        #find atom that has not been assigned the type
        if not isinstance(atomTypes[i],str):
            bondedId = np.where(bonds[i] > 0.)[0].tolist() #index of atoms that the current bonded to
            bondedAtoms  = []
            for Id in bondedId: 
                bondedAtoms.append(atomTypes[Id])
            if at == atomTypeDict['HA'] and 'CA' in bondedAtoms:
                atomTypes[i] = 'HA' 
            elif at == atomTypeDict['HC'] and 'CC' in bondedAtoms:
                atomTypes[i] = 'HC'
    #check
    if not all(isinstance(at,str) for at in atomTypes):
        raise Exception('Not all atoms have been assigned type')
    return atomTypes
                        
def getAvgCharge(ac, charges, atomTypes, atomidBoundsatomidBounds,resID = None, prepgen = False, ext = ''):
    #modify file name depends on whether charges is generated with or without adjustment using prepgen
    if not prepgen:
        ext = "_avgCharge" + ext + "_noPrepgen"
    else:
        ext = "_avgCharge" + ext + "_Prepgen"
    fOut = ac.split('.ac')[0] + ext + ".txt"   
    indexDict = {'CA': [], 'CB': [], 'CC': [],
                     'O': [], 'OH': [],
                     'HA': [], 'HC': [], 'HO': []} #indices according to atom types

    avgChargeDict = {'CA': [], 'CB': [], 'CC': [],
                     'O': [], 'OH': [],
                     'HA': [], 'HC': [], 'HO': []} #avg and std of charges for each type
    nAtomPerType =  {'CA': [], 'CB': [], 'CC': [],
                     'O': [], 'OH': [],
                     'HA': [], 'HC': [], 'HO': []}  

    #modify charges and atomTypes list to only contain atoms in residues specified by resID
    if not resID == None:
        atomIDs = []
        inRes = []
        for i in resID:
            atomIDs += list(range(atomidBoundsatomidBounds[i][0]-1,atomidBoundsatomidBounds[i][1]))
        for i in range(len(charges)):
            inRes.append(i in atomIDs)
        inRes = np.array(inRes, dtype = bool)
        charges = np.array(charges)[inRes]
        atomTypes = np.array(atomTypes)[inRes] 
    #get indices of all atom types
    totalCharge = 0
    for key in indexDict.keys():
        indexDict[key] = np.where(np.array(atomTypes, dtype = str) == key)[0].tolist()
    #calculate average charges
    for key in  avgChargeDict.keys():
        ind = indexDict[key]
        n = float(len(ind)) #number of this atom type
        charges_temp = []
        nAtomPerType[key] = n
        for j in ind:
            charges_temp.append(charges[j])
        try:
            avg = sum(charges_temp) / n
            std = np.std(charges_temp)
        except:
            avg = 'N/A'
            std = 'N/A'
        avgChargeDict[key] = [avg, std]
        totalCharge += sum(charges_temp)
        
    print ('Total charge in ac file: %5.5f'%totalCharge)
    #check whether all charges were used in calculate average charges
    if abs(totalCharge - sum(charges)) > 10.e-5:
        raise Exception('Total charges do not match')
    #write out to file
    s = "#AtomType AverageCharge stdev %stdev numberOfThisType\n"
    for key, value in sorted(avgChargeDict.items()):
        if not isinstance(value[0],str): #dont write out if an atom type is not in the residue
            try:
                percentStd = abs(value[1]/value[0]) * 100.
                s += "%s %5.6f %5.6f %5.2f %i\n" %(key, value[0], value[1], percentStd, nAtomPerType[key])
            except:
                percentStd = 'N/A'
                s += "%s %5.6f %5.6f %s %i\n" %(key, value[0], value[1], percentStd, nAtomPerType[key])
    s += "#Total charge of all atoms: %5.7f\n" %sum(charges)
    f = open(fOut,'w')
    f.write(s)    
    return(avgChargeDict)  

def runPrepgen(atomNames, atomTypes, mainChainTypes, nRes, ac, nAtomMainChain, atomidBounds, resCharges):
    """write .mc file to truncate the molecule into segments of residues
    and run prepgen"""
    isMainChain = MainChain(atomTypes, mainChainTypes)
    
    mainChain = np.array(atomNames)[isMainChain]
    print ('Atoms in main chain \n{}'.format(mainChain))
    #append string defining the chain
    s = ''
    s += 'HEAD_NAME {}\n'.format(mainChain[0])
    s += 'TAIL_NAME {}\n'.format(mainChain[-1])
    for aname in mainChain[1:-1]:
        s += 'MAIN_CHAIN {}\n'.format(aname)
    prepins = []
    removedHC = False
    removedHA = False
    print('\nrunning prepgen ...')          
    for res in range(nRes):
        resCharge = resCharges[res]
        lowBound = atomidBounds[res][0]
        hiBound = atomidBounds[res][-1]
        atomNamesPerRes = [aname for i, aname in enumerate(atomNames) if (i+1) >= lowBound and (i+1) <= hiBound ]
        s1 = ''
        for i, atName in enumerate(atomNames):
            if atName not in atomNamesPerRes:
                s1 += 'OMIT_NAME {}\n'.format(atName)
            else:
                if res == 0 and atomTypes[i] == 'HA' and not removedHA:
                    s1 += 'OMIT_NAME {}\n'.format(atName)
                    removedHA = True
                elif res == nRes-1 and atomTypes[i] == 'HC' and not removedHC:
                    s1 += 'OMIT_NAME {}\n'.format(atName)
                    removedHC = True
                
        s1 += """PRE_HEAD_TYPE C
POST_TAIL_TYPE C
CHARGE %2.2f""" %resCharge
        s2 = s + s1
        Prefix = ac.split('.ac')[0] + '_res%i'%(res+1)
        mc = Prefix + '.mc'
        file = open(mc,'w')
        file.write(s2)
        file.close()
        
        #running prepgen
        prepin = Prefix + '.prepin'
        os.system('prepgen -i {ac} -o {Prefix}.prepin -m {mc}'.format(ac = ac, Prefix = Prefix, mc = mc))
        print('\nprepgen -i {ac} -o {prepin} -m {mc}'.format(ac = ac, prepin = prepin, mc = mc))
        
        #reading outputted prepin
        while not os.path.exists(prepin):
            time.sleep(1)
        prepins.append(prepin) 
    charges = readPrepin(prepins, atomNames, atomTypes, resCharges) 
    return charges
                
def readPrepin(prepins, atomNames, atomTypes, resCharges):
    """read prepin files and create an array of charges for each atom"""
    charges = np.zeros(len(atomNames)) 
    for i, prepin in enumerate(prepins):
        chargePerRes = []
        print('reading charges from {}'.format(prepin))
        file = open(prepin, 'r')
        lines = file.readlines()
        file.close()
        for line in lines:
            try:
                cols = line.split()
                int(cols[0])
                if len(cols) > 3:
                    atomName = cols[1]
                    #only append charges if the atom name in prepin file match the atom names in atomNames (exclude dummy atoms)
                    if atomName in atomNames:
                        charge = float(cols[-1])
                        chargePerRes.append(charge)
                        #get index of this atom in the list of all atom names and append charge
                        ind = atomNames.index(atomName)
                        charges[ind] = charge
            except:
                pass   
            #check if total charges of the residue generated by prepgen match the target
        if abs(sum(chargePerRes) - resCharges[i]) > 1.e-4:
            raise Exception('Total charges in prepin does not match the target charge for residue %i'%(i+1))
    return charges
    
def MainChain(atomTypes, mainChainTypes):
    isMainChain = []
    for at in atomTypes:
        isMainChain.append(at in mainChainTypes)
    isMainChain = np.array(isMainChain, dtype = bool)
    return isMainChain
   
#def runPrepgen():
    
if __name__ == "__main__":
    #INPUTS
    ac = 'AA7_f0.57.ac'
    #list of number of atoms in each residue 
    nAtomList = [9,9,8,9,8,9,9] 
    nRes = len(nAtomList)
    #charges of each residue to use in .mc file (should be integer charges)
    resCharges = [-1.,0.,-1.,0.,-1.,0.,-1]
    #residue ids to perform averaging (only used if having mixed type of residues)
    resID = [0,2,4,6] 
    #these atom types is the carbon backbone
    mainChainTypes = ['CA', 'CC']
    #number of atoms per residue on main chain
    nAtomMainChain = 2    
    #possible atom types for each element:
    Ctypes = ['c', 'c3']
    Htypes = ['h', 'ho', 'hc']
    Otypes = ['o', 'oh']
    atomTypeDict = {'CA': 'c3', 'CB': 'c', 'CC': 'c3',
                    'O': 'o', 'OH': 'oh', 'HA': 'hc', 'HC': 'hc',
                    'HO': 'ho'} # atomTypes : atomTypesGen
    if len(resCharges) != len(nAtomList):
        raise Exception('mismatch in dimension of resCharges and nAtomList')
        
    ##running script---    
    posInd0, bondInd0, nAtomTot, atomidBounds, charges, atomTypesGen, bonds, atomNames, ac = readAC(ac, nAtomList, Ctypes, Htypes, Otypes)
    atomTypes = getAtomType(bonds, nAtomTot, atomTypeDict, atomTypesGen)
    charges = runPrepgen(atomNames, atomTypes, mainChainTypes, nRes, ac, nAtomMainChain, atomidBounds, resCharges)

    #check whether having mixed residue types
    if all (x == resCharges[0] for x in resCharges):
        avgChargeDict = getAvgCharge(ac, charges, atomTypes,atomidBounds, prepgen = False)
        charges = runPrepgen(atomNames, atomTypes, mainChainTypes, nRes, ac, nAtomMainChain, atomidBounds, resCharges)
        #averaging over charges of all residues
        print('\nAveraging charges over all residues')
        avgChargeDict = getAvgCharge(ac, charges, atomTypes, atomidBounds,prepgen = True, ext = 'AllRes')
        #averaging over charges of  interior residues
        print('\nAveraging charges over only interior residues')
        resID = range(1,nRes-1)
        avgChargeDict = getAvgCharge(ac, charges, atomTypes, atomidBounds, resID = resID, prepgen = True, ext = 'interior')

        #charge of middle mer
        if len(nAtomList)%2 != 0:
          indMiddle = int(np.ceil(float(len(nAtomList))/2.))-1
          print('\nGetting charge of the middle monomer, monomer %i' %(indMiddle+1))
          avgChargeDict = getAvgCharge(ac, charges, atomTypes, atomidBounds, resID = [indMiddle], prepgen = True, ext = 'middle') 
    else:
        print('\nAveraging charges over residues {}'.format(resID))
        avgChargeDict = getAvgCharge(ac, charges, atomTypes, atomidBounds, resID = resID, prepgen = True, ext = 'DeprotRes'.format(resID))

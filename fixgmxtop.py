#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:09:34 2019

@author: nvthaomy
"""
"""Fix the function type of angles from 1 to 5, proper dihedral function to 9
    and add improper dihedrals to gromacs topology file"""
def main(inFile,outFile):
    section = 0
    angle_f = '5'
    proper_dihedral_f = '9'
    improper_dihedral_f = '2'
    writing_dihedral = []
    counter = 0
    improp_dihedral = []
    CA=[]
    CB=[]
    O1=[]
    O2=[]
    residue = 0
    infile = open(inFile,'r')
    line = infile.readline()
    while len(line) or not section == 'done':
        if line.startswith('[ atoms ]'):
            section = 'atoms'
        if section == 'atoms':
            if '; residue' in line: #mark when move on to the next residue
                counter += 1
                if O2:
                    if residue == 'AHD' or residue == 'AD' or residue == 'ATD':
                        improper ='\t'.join([CB,O1,O2,CA,improper_dihedral_f,'\n'])
                        improp_dihedral.append(improper) 
                    else: #sequence of atoms in improper is different for two versions of protonations
                        improper ='\t'.join([CB,CA,O1,O2,improper_dihedral_f,'\n'])
                        improp_dihedral.append(improper)
                residue = line.split()[3]
            elif not (line.startswith(';') or line.startswith('[')) and len(line.split()) != 0:
                if line.split()[4] == 'CB':#storing id of carbonyl C
                    CB = line.split()[0]
                elif line.split()[4] == 'OD1':
                    O1 = line.split()[0]
                elif line.split()[4] == 'OD2':
                    O2 = line.split()[0]
                elif line.split()[4] == 'CA':
                    CA = line.split()[0] 
            elif len(line.split())==0:
                if residue == 'AHD' or residue == 'AD' or residue == 'ATD':
                        improper ='\t'.join([CB,O1,O2,CA,improper_dihedral_f,'\n'])
                        improp_dihedral.append(improper) 
                else: #sequence of atoms in improper is different for two versions of protonations
                        improper ='\t'.join([CB,CA,O1,O2,improper_dihedral_f,'\n'])
                        improp_dihedral.append(improper)
                section = 'done'
        line = infile.readline()
    infile.close()
                    
    with open(outFile,'w') as outfile:
        with open(inFile,'r') as infile:
            line = infile.readline()
            while len(line):
                if line.startswith('[ angles ]'):
                    section = 'angles'
                elif line.startswith('[ dihedrals ]'):
                    section = 'dihedrals'
                if section == 'angles' and not line.startswith('[') and not line.startswith(';') and not len(line.split())==0 :
                    newline = line.split()
                    newline[3] = angle_f
                    newline.append('\n')
                    newline ='\t'.join(newline)
                    outfile.write(newline)
                elif section == 'dihedrals' and not line.startswith('[') and not line.startswith(';') and not len(line.split())==0:
                    newline = line.split()
                    newline[4] = proper_dihedral_f
                    newline.append('\n')
                    newline = '\t'.join(newline)
                    outfile.write(newline)
                    writing_dihedral = True
                elif section == 'dihedrals' and writing_dihedral and len(line.split())==0:
                    outfile.write(line)
                    outfile.write('[ dihedrals ]\n')
                    outfile.write(';improper dihedrals\n')
                    for i in improp_dihedral:
                        outfile.write(i)
                    outfile.write('\n')
                    section = 'done'
                else:
                    outfile.write(line)
                line = infile.readline()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,help=".top file")                 
    parser.add_argument("-o", "--output", help="output file")
    args = parser.parse_args()
    infile = args.input
    outfile = args.output
    main(infile,outfile)                
                
            
                
                
                
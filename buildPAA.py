#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:56:07 2019

@author: nvthaomy
"""
import random
import numpy as np
import math
import os

""" write out tleap input file to build a single PAA chain with specified
    tacticity and various fraction of deprotonation
    input: N: degree of polymerization
           f: fraction of deprotonation
           Pm: fraction of meso diads"""
class Molecule:
    def __init__(self,N,f,Pm):
        self.DOP = N
        self.charge = f #a list of deprotonation fraction e.g. [0,0.2,0.6,1]
        self.Pm = Pm
    def get_indices(self,num1,num2,N):
        """get indices of deprotonated monomers (for f<=0.5) or protonated monomer
           (for f>0.5) in a chain
           N: degree of polymerization
           num2: number of deprotonated monomers (for f<=0.5) or protonated monomer (for f>0.5)
           num1: list of upper and lower bounds of number of protonated monomers if f  <= 0.5 or 
                number of deprotonated monomers if f > 0.5
           output: list of indices"""
        i=[] #list of indices of deprotonated monomers
        num2 = (round(num2))
        if len(num1) > 1: #for case when num is not an integer or divisible by 5
            i.append(random.randint(0,min(num1)))
            print "place the first deprot (or prot if f>0.5) monomer at position %i" %(i[0]+1)
            counter = 0 
            old_j = 0
            for j in range(1,int(num2)):
                spacing = num1[random.randint(0,len(num1)-1)] #draw random spacing from the list of upper and lower bounds of number of protonated(deprotonated) monomers 
                i.append(i[j-1] + spacing + 1)
                while i[j] > N-1:
                    i[j] -= 1        
                    if j != old_j: #only count if the next monomer index exceed the monomer chain length
                        counter += 1 
                        if counter > 1:
                            print "\nWarning: Index of monomer is out of range!"
                    old_j = j 
        else:
            i.append(0)
            counter = 0
            old_j = 0
            print "place the first deprot/prot monomer at position %i" %(i[0]+1)
            for j in range(1,int(num2)):
                i.append(i[j-1] + num1[0] + 1)
                while i[j] > N-1:
                        i[j] -= 1        
                        if j != old_j: #only count if the next monomer index exceed the monomer chain length
                            counter += 1 
                            if counter > 1:
                                print "\nWarning: Index of monomer is out of range!"
                        old_j = j 
        i = [int(x) for x in i] #convert indices to integer
        return i
            
    def uniform_charge_pattern(self,charge_frac):
        print "\nGenerating evenly distributed charge pattern for f = %3.2f" %charge_frac 
        N=self.DOP
        f0= N*['p']
        f1= N*['d']
        
        if charge_frac == 0:
            charge_pattern = f0
        if charge_frac == 1:
            charge_pattern = f1
        if charge_frac != 0 and charge_frac != 1:
            print "\nDeprotonation fraction is not 0 or 1"
            
            if charge_frac <= 0.5:
                num_deprot = (charge_frac * N)
                num_prot = [(N/num_deprot - 1)] 
                print "\nNumber of protonated monomers between each deprotonated monomer is: %3.3f" %num_prot[0]
                if not abs(num_prot[0]-round(num_prot[0])) < 0.0001: #not an integer
                    if num_prot[0]*10 % 5 == 0: #screen out charge fraction that result in num_prot= x.5
                        num_prot = [math.ceil(num_prot[0])]
                    else:
                        num_min = math.floor(num_prot[0])
                        num_max = math.ceil(num_prot[0])
                        mean = num_prot[0] #average number of protonated moners in between two deprotonated ones
                        num_prot = [num_min]
                        calc_mean = sum(num_prot)/len(num_prot)
                        
                        while abs(calc_mean - mean) > 0.001:
                            if calc_mean > mean:
                                num_prot.append(num_min)
                                calc_mean = sum(num_prot)/len(num_prot)
                            else:
                                num_prot.append(num_max)
                                calc_mean = sum(num_prot)/len(num_prot)
                        print calc_mean
                        print ('\nNunmber of protonated monomers will be drawn from this list:')
                        print num_prot
                
                i_deprot = self.get_indices(num_prot,num_deprot,N)
                print "\n Indices of deprotonated monomer for f = %3.2f are " %charge_frac
                print i_deprot
                if abs(len(i_deprot)-num_deprot) > 0.001:
                    print "Number of deprotonated monomers does not match the value of charge fraction!\n"
                charge_pattern = f0
                for i in i_deprot:
                    charge_pattern[i] = 'd'
            
            if charge_frac > 0.5:
                num_prot = ((1 - charge_frac) * N)
                num_deprot = [(N/num_prot - 1)]
                print "\nNumber of deprotonated monomers between each protonated monomer is: %3.3f" %num_deprot[0]
                if not abs(num_deprot[0]-round(num_deprot[0])) < 0.0001: #not an integer
                    if num_deprot[0]*10 % 5 == 0: #screen out charge fraction that result in num_deprot= x.5
                        num_deprot = [math.ceil(num_deprot[0])]
                    else:
                        num_min = math.floor(num_deprot[0])
                        num_max = math.ceil(num_deprot[0])
                        mean = num_deprot[0] #average number of deprotonated moners in between two protonated ones
                        num_deprot = [num_min]
                        calc_mean = sum(num_deprot)/len(num_deprot)
                        
                        while abs(calc_mean - mean) > 0.001:
                            if calc_mean > mean:
                                num_deprot.append(num_min)
                                calc_mean = sum(num_deprot)/len(num_deprot)
                            else:
                                num_deprot.append(num_max)
                                calc_mean = sum(num_deprot)/len(num_deprot)
                        print calc_mean
                        print ('\nNunmber of deprotonated monomers will be drawn from this list:')
                        print num_deprot
                print('num_deprot {}'.format(num_deprot))
                print('num_prot {}'.format(num_prot))
                i_prot = self.get_indices(num_deprot,num_prot,N)
                print "\nIndices of protonated monomer for f = %3.2f are " %charge_frac
                print i_prot
                if abs(len(i_prot)-num_prot) > 0.001:
                    Exception("Number of protonated monomers does not match the value of charge fraction!\n")
                charge_pattern = f1
                for i in i_prot:
                    charge_pattern[i] = 'p' 
        return charge_pattern
    
    def charge_pattern(self,charge_frac,pattern):
        """Evaluate if charge pattern is random or evenly distributed
        and enerate charge pattern"""
        if pattern == 'random':
            print "\nGenerating randomly distributed charge pattern for f = %3.2f" %charge_frac 
            N=self.DOP
            f0= N*['p']
            f1= N*['d']
            num_deprot = N*charge_frac
            if charge_frac == 0:
                charge_pattern = f0
            if charge_frac == 1:
                charge_pattern = f1
            else:
                charge_pattern = f0
                i_deprot = random.sample(range(N),int(num_deprot))
                print i_deprot
                for i in i_deprot:
                    charge_pattern[i] = 'd' 
            print charge_pattern
        else:
            charge_pattern = self.uniform_charge_pattern(charge_frac)
        return charge_pattern           
                
        
    def join_tact_charge(self,charge_frac,tacticity,pattern):
        """Append tacticity with charge pattern"""
        tact_charge=[]
        charge_pattern = self.charge_pattern(charge_frac,pattern)
        for i in range(0,self.DOP):
            a = tacticity[i] + charge_pattern[i]
            tact_charge.append(a)
        tact_charge[0]=tact_charge[0].upper()
        tact_charge[-1]=tact_charge[-1].upper()
        return tact_charge
        
    def tacticity(self,Pm):
            N=self.DOP
            dyad=[]
            tact=[]
            #follow Bernoullian statistics (common with free radical polymerization)
            #stereo of the next monomer is independent of the stereochemistry of growing chain
            for i in range(0,N):
                rand = np.random.random()
                if rand <= Pm:
                        dyad.append('m')
                else:
                        dyad.append('r')
            print('Diad sequence from meso diad fraction of {}:\n{}'.format(Pm,dyad))
            tact.append('AH') #head group is achiral
            tact.append('u') #arbitrarily pick the second monomer to be "up"
            for i in range(1,len(dyad)-2):
                if dyad[i] == 'm':
                    tact.append(tact[i])
                else:
                    if tact[i] == 'u':
                        tact.append('d')
                    else:
                        tact.append('u')
            tact.append('AT')
            return tact
def buildPAA(f,N,Pm,pattern):
    with open("build_AA"+str(N)+".log", 'w') as log:     
        PAA=Molecule(N,f,Pm) #f is a list of deprotonation fraction
        tact_charge_matrix=[] #each row is the charge pattern of each protonation fraction
        log.write("\nCalculating number of deprotonated monomers and"\
                  "modifying input fraction of deprotonation:")        
        for index,charge_frac in enumerate(f):
            if charge_frac !=0:
                num_deprot = float(charge_frac * N)
                if not num_deprot.is_integer(): 
                    log.write("\nNeed to modify f = %3.2f to " %charge_frac)
                    num_deprot = round(num_deprot)
                    charge_frac = num_deprot/N #new charge fraction
                    log.write("f = %3.2f" %charge_frac)
                    f[index] = charge_frac
        log.write("\nNew charge vector is :"%f)
        rounded_charge = [round(charge,2) for charge in f]
        log.write('{}'.format(rounded_charge))
        log.write('\nBuilding PAA of with N = {}, f = {}, charge pattern = {},'\
                  ' meso fraction  = {}'.format(N,rounded_charge,pattern,Pm))
        tacticity=PAA.tacticity(Pm)
        for index,charge_frac in enumerate(f):
            tact_charge = PAA.join_tact_charge(charge_frac,tacticity,pattern)
            tact_charge_matrix.append(tact_charge)
        
        log.write("\nWriting tleap input file to build a single polymer with different deprotonated fraction")
        file = open("build_AA"+str(N)+".in","w")
        file.write("source leaprc.gaff2\n")
        file.write("loadOFF PAA.lib\n")
        file.write("up =loadpdb AA_prot1.pdb\n")
        file.write("ud =loadpdb AA_deprot1.pdb\n")
        file.write("dp =loadpdb AA_prot2.pdb\n")
        file.write("dd =loadpdb AA_deprot2.pdb\n")
        file.write("set up head up.1.1\n")
        file.write("set up tail up.1.3\n")
        file.write("set ud head ud.1.1\n")
        file.write("set ud tail ud.1.3\n")
        file.write("set dp head dp.1.1\n")
        file.write("set dp tail dp.1.3\n")
        file.write("set dd head dd.1.1\n")
        file.write("set dd tail dd.1.3\n")
        file.write("\n")
        pdbList = []
        newCharge = []
        for index,charge_frac in enumerate(f):
            if abs(charge_frac - 0) < 10**(-3) or abs(charge_frac - 1) < 10**(-3):
                charge_frac = int(charge_frac)
            else:
                charge_frac = round(charge_frac,2)
            newCharge.append(charge_frac)
            sequence =' '.join(tact_charge_matrix[index])
            file.write('#f = {}\n'.format(round(charge_frac,2)))
            file.write("x = sequence{")
            file.write("{}".format(sequence))
            file.write("}\n")
            file.write("savepdb x AA{}_f{}.pdb\n".format(N,charge_frac))
            file.write("\n")  
            pdbList.append('AA{}_f{}.pdb'.format(N,charge_frac))
        file.write("quit")
        file.close()
        log.write("\nDone writing tleap input file")
    log.close()
    build_file = "build_AA"+str(N)+".in"
    return build_file,pdbList,newCharge
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",nargs='+', type=float, required=True,
                        help="list of deprotonation fractions e.g. 0 0.2 0.5 1")
    parser.add_argument("-N",type=int, required=True,
                        help="degree of polymerization")
    parser.add_argument("-Pm","--Pm",type=float,default = 1, 
                        help="Meso diad fraction, isotactic if = 1, syndiotactic if = 0")
    parser.add_argument("-r","--random", action = "store_true",
                        help="Random deprotonation, default pattern is random")
    parser.add_argument("-e","--evendist", action = "store_true",
                        help="Evenly distributed deprotonation")
    args = parser.parse_args()    
    f = args.f
    N = args.N
    Pm = args.Pm
    pattern = 'random' #random deprotonation
    if args.evendist:
        pattern = 'even'
    buildPAA(f,N,Pm,pattern)


     




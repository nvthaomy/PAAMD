#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:06:15 2019

@author: nvthaomy
"""

def loadFF(watermodel,mixturePdb,lib):
    """write tleap input file to load forcefield and generate .parm7 and .crd
        mixture Pdb is list of all mixture pdb files
	lib: name of tleap library for PAA monomers"""
    with open('loadFF.in','w') as load:
        load.write('source leaprc.gaff2')
        load.write('\nsource leaprc.water.{}'.format(watermodel))
        for i in lib:
            load.write('\nloadOFF {}.lib\n'.format(i))
        topFile = []
        crdFile = []
        for pdb in mixturePdb:
            topname = pdb[:pdb.index('w')]+'gaff2_'+pdb[pdb.index('w0'):pdb.index('pdb')]+'parm7'
            crdname = pdb[:pdb.index('pdb')]+'crd'
            topFile.append(topname)
            crdFile.append(crdname)
            load.write('\n\nx=loadpdb {}'.format(pdb))
            load.write('\naddions x Na+ 0')
            load.write('\nsetbox x vdw 1')
            load.write('\nsaveamberparm x {} {}'.format(topname,crdname))
	    load.write('\nsavepdb x {}.pdb'.format(topname.split('.parm7')[0]))
        load.write('\nquit') 
    return 'loadFF.in',topFile,crdFile
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",nargs='+', required=True,
                        help="list of mixture pdb")
    parser.add_argument("-w","--watermodel",
                        help="Water model for simulation (opc,tip3p,spce), default = opc")
    parser.add_argument("-l", nargs='+', 
			help="tleap library for PAA monomers: PAA, PAA_avg, PAA1, etc.")
    args = parser.parse_args() 
    if args.watermodel:
        watermodel = args.watermodel
    else:
        watermodel = 'opc'
    mixturePdb = args.p
    watermodel = args.w
    loadFF(watermodel,mixturePdb,args.l)

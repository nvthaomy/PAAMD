import stats_openmm as stats
import numpy as np
import matplotlib, sys, os
import matplotlib.pyplot as plt
import MDAnalysis as mda
import mdtraj as md
sys.path.append('/home/mnguyen/bin/scripts/')
import log2txt
showPlots = True
try:
    os.environ["DISPLAY"] #Detects if display is available
except KeyError:
    showPlots = False
    matplotlib.use('Agg') #Need to set this so doesn't try (and fail) to open interactive graphics window


autowarmup = True
#defulat warmup samples if not using autowarmup
warmup = 100

trajFiles = ['trajectory298.dcd']
tops = ['AA12_f0.25_opc_gaff2_w0.13.parm7']
stride = 5

#names of residue in polymer chain
#resInChain = ['AHP','AP', 'ATP']
DOPs = [12]
NPs = [15]
#index of first polymer residue
res0Id = 0


#########################End of input######################
def getThermo(ThermoLog, fi = 'lammps', obs = None, cols = None, autowarmup = True, warmup = 100):
    """ fi: log file format, 'lammps' or 'openmm' """
    if not obs == None and not cols == None:
        Exception('Read data either by observable name or column index but not both!')

    #conver log file:
    if fi == 'openmm':
            ThermoLog = log2txt.log2txt_openmm([ThermoLog])[0]
    elif fi == 'lammps':
            section = 'PRODUCTION RUNS'
            ThermoLog = log2txt.log2txt_lammps([ThermoLog],section,'production')[0]

    print('new log file: {}'.format(ThermoLog))
    txt = ""
    obsID = []
    Stats = []
    #do stats
    file = open(ThermoLog,'r')
    if not obs == None:
        lines = file.readlines()
        while not isinstance(cols,list):
            for line in lines:
                if line.startswith('#'):
                                   obsNames = line.split()[1:]
                                   print('obsNames {}'.format(obsNames))
                                   cols = [obsNames.index(val) for val in obsNames if val in obs]
    print('cols {}'.format(cols))
    for i, col in enumerate(cols):
        if autowarmup:
            warmup,Data,nwarmup = stats.autoWarmupMSER(file, col)
            print ("Auto warmup detection with MSER-5 => ",nwarmup)
        else:
            warmup,Data = stats.extractData(file, col, warmup)
        (nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor)=stats.doStats(warmup,Data, False ,False,'_{0}_mol{1}'.format(file.name,col))
        try:
            obsName = obsNames[col]
        except:
            obsName = 'col{}'.format(col)
        lines = "" 
        lines += '\n==== {} ===='.format(obsName)
        lines += "\n  - Mean                    = {} +/- {}".format(mean,semcc)
        lines += "\n  - Equilibrated samples    = {}".format(nsamples)
        lines += "\n  - Correlation time        = {}".format(kappa)
        lines += "\n  - Effective # samples     = {}".format(nsamples/kappa)
        lines += "\n  - Reduced-bias variance   = {}".format(unbiasedvar)
        # note that there is no unbiased estimator for the population standard deviation. We can use sqrt(var) as a indicative estimator.
        lines += "\n  - S.D. (unbiased, biased) = {} {}".format(np.sqrt(unbiasedvar),np.std(Data,ddof=0)) # ddof is correction to 1/N...using ddof=1 returns regular reduced-bias estimator
        lines += "\n  - Min, Max                = {} {}\n".format(min,max)

        print(lines)
        txt += lines
    
        Avg = mean
        Std = np.sqrt(unbiasedvar)
        Err = semcc
        CorrTime = kappa 
        NUncorrSamples = nsamples/kappa
        Stats.append([Avg,Std,CorrTime,Err,NUncorrSamples])
        obsID.append(obsName)

    return obsID, Stats
        
def getRgRee(trajFile, top, DOP, NP, NAtomsPerChain = None, 
             RgDatName = 'RgTimeSeries', ReeDatName = 'ReeTimeSeries',RgStatOutName = 'RgReeStats', Ext='.dat',  
             res0Id = 0, stride = 1, autowarmup = True, warmup = 100, plot = False):    
    
    """NAtomsPerChain: used if running CG system, if provided will assume there is one residue per chain"""
    ElementDictionary ={
                        "carbon": 12.01,
                        "hydrogen": 1.008,
                        "oxygen": 16.00,
                        "nitrogen": 14.001,
                        "virtual site": 1.0,
                        "virtual_site": 1.0,
                        "sodium": "na+"}

    traj = md.load(trajFile, top=top, stride = stride)
#    traj.make_molecules_whole(inplace=True, sorted_bonds=None) # Automatically finds the bonds from the topology file

    RgStats = []
    RgTimeseries = [range(traj.n_frames)]
    Rgheader = "Frame   "
    txtRg = ""
    
    ReeStats = []
    ReeTimeseries = [range(traj.n_frames)]
    Reeheader = "Frame   "
    
    #get indices of residues in all chains    
    MoleculeResidueList = []
    if not NAtomsPerChain:
        #number residues per chain = DOP (for AA systems)
        for j in range(NP):
            resId = range(res0Id + j*DOP, res0Id + (j+1)*DOP)
            MoleculeResidueList.append(resId)
    else:
        #1 residue per chain (for CG system)
        x = range(res0Id, res0Id + NP)
        MoleculeResidueList = [[a] for a in x]
        
    for j,resId in enumerate(MoleculeResidueList):
        resIdLow = np.min(resId)
        resIdUp = np.max(resId)
        atom_indices = traj.topology.select('resid {} to {}'.format(resIdLow,resIdUp)) 
        print('Indices of atoms in chain {} \n{}'.format(j+1,atom_indices))
        mass_list = []
        for index in atom_indices:
            element = str(traj.topology.atom(index).element)
            try:
                mass = ElementDictionary[element]
            except:
                mass = 1.
            mass_list.append(mass)
        mass_list = np.array(mass_list)
        
        '''=== Compute Rg ==='''
        Rg = md.compute_rg(traj.atom_slice(atom_indices),masses=mass_list) 
        RgTimeseries.append(Rg.tolist())
        Rgheader += 'Rg{}   '.format(j+1)
        np.savetxt(RgDatName+Ext, np.transpose(RgTimeseries), fmt = '%5.5f', header=Rgheader ) 
        
        
        #do stats
        file = open(RgDatName+Ext,'r')
        if autowarmup:
            warmup,Data,nwarmup = stats.autoWarmupMSER(file, j+1)
            print ("Auto warmup detection with MSER-5 => ",nwarmup)
        else:
            warmup,Data = stats.extractData(file, j+1, warmup)
        (nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor)=stats.doStats(warmup,Data, False ,False,'_{0}_mol{1}'.format(file.name,j+1))

        lines = "" 
        lines += '\n==== Rg for molecule {} ===='.format(j+1)
        lines += "\n  - Mean                    = {} +/- {}".format(mean,semcc)
        lines += "\n  - Equilibrated samples    = {}".format(nsamples)
        lines += "\n  - Correlation time        = {}".format(kappa)
        lines += "\n  - Effective # samples     = {}".format(nsamples/kappa)
        lines += "\n  - Reduced-bias variance   = {}".format(unbiasedvar)
        # note that there is no unbiased estimator for the population standard deviation. We can use sqrt(var) as a indicative estimator.
        lines += "\n  - S.D. (unbiased, biased) = {} {}".format(np.sqrt(unbiasedvar),np.std(Data,ddof=0)) # ddof is correction to 1/N...using ddof=1 returns regular reduced-bias estimator
        lines += "\n  - Min, Max                = {} {}\n".format(min,max)
        print(lines)
        txtRg += lines

        RgAvg = mean
        RgStd = np.sqrt(unbiasedvar)
        RgErr = semcc
        CorrTime = kappa 
        NUncorrSamples = nsamples/kappa
        RgStats.append([RgAvg,RgStd,CorrTime,RgErr,NUncorrSamples])

#        print ('The Rg for molecule {} (mean, error, std)'.format(j))
#        print ('\t{0:2.4f}\t{1:2.5f}\t{1:2.5f}'.format(RgAvg, RgErr, RgStd))

        ''' Plot Rg '''
        if plot:
            plt.plot(Rg, "k-")
            plt.xlabel('timestep')
            plt.ylabel('Radius-of-gryation')
            plt.savefig("Rg{}.png".format(j+1),bbox_inches='tight')
            plt.close()

        '''=== Compute Ree ==='''
        atom_pairs = [np.min(atom_indices), np.max(atom_indices)]
        Ree = md.compute_distances(traj,atom_pairs= [atom_pairs], periodic=False, opt=True)
        Ree = Ree.tolist()
        Ree = [a[0] for a in Ree]
        ReeTimeseries.append(Ree)
        Reeheader += 'Ree{}   '.format(j+1)
        np.savetxt(ReeDatName+Ext, np.transpose(ReeTimeseries), fmt = '%5.5f', header=Reeheader ) 
               
        #do stats
        file = open(ReeDatName+Ext,'r')
        if autowarmup:
            warmup,Data,nwarmup = stats.autoWarmupMSER(file, j+1)
            print ("Auto warmup detection with MSER-5 => ",nwarmup)
        else:
            warmup,Data = stats.extractData(file, j+1, warmup)
        (nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor)=stats.doStats(warmup,Data, False ,False,'_{0}_mol{1}'.format(file.name,j+1))

        lines = "" 
        lines += '\n==== Ree for molecule {} ===='.format(j+1)
        lines += "\n  - Mean                    = {} +/- {}".format(mean,semcc)
        lines += "\n  - Equilibrated samples    = {}".format(nsamples)
        lines += "\n  - Correlation time        = {}".format(kappa)
        lines += "\n  - Effective # samples     = {}".format(nsamples/kappa)
        lines += "\n  - Reduced-bias variance   = {}".format(unbiasedvar)
        # note that there is no unbiased estimator for the population standard deviation. We can use sqrt(var) as a indicative estimator.
        lines += "\n  - S.D. (unbiased, biased) = {} {}".format(np.sqrt(unbiasedvar),np.std(Data,ddof=0)) # ddof is correction to 1/N...using ddof=1 returns regular reduced-bias estimator
        lines += "\n  - Min, Max                = {} {}\n".format(min,max)
        print(lines)
        txtRg += lines

        ReeAvg = mean
        ReeStd = np.sqrt(unbiasedvar)
        ReeErr = semcc
        CorrTime = kappa 
        NUncorrSamples = nsamples/kappa
        ReeStats.append([ReeAvg,ReeStd,CorrTime,ReeErr,NUncorrSamples])

        ''' Plot Ree '''
        if plot:
            plt.plot(Ree, "k-")
            plt.xlabel('timestep')
            plt.ylabel('End-to-end distance')
            plt.savefig("Ree{}.png".format(j+1),bbox_inches='tight')
            plt.close()

    #get averages of stats
    RgStats = np.array(RgStats)
    RgAvg = np.mean(RgStats[:,0])
    RgStd = np.mean(RgStats[:,1])
    RgCorrTime = np.mean(RgStats[:,2])
    RgErr = np.mean(RgStats[:,3])
    RgErr_Prop = np.sqrt(np.sum(RgStats[:,3]**2))/NP
    RgCorrTimeErr = np.sqrt(np.var(RgStats[:,2])/len(RgStats[:,2]))
    RgNUncorrSamples = np.mean(RgStats[:,4])

    ReeStats = np.array(ReeStats)
    ReeAvg = np.mean(ReeStats[:,0])
    ReeStd = np.mean(ReeStats[:,1])
    ReeCorrTime = np.mean(ReeStats[:,2])
    ReeErr = np.mean(ReeStats[:,3])
    ReeErr_Prop = np.sqrt(np.sum(ReeStats[:,3]**2))/NP
    ReeCorrTimeErr = np.sqrt(np.var(ReeStats[:,2])/len(ReeStats[:,2]))
    ReeNUncorrSamples = np.mean(ReeStats[:,4])

    lines = ""
    lines += '\n\n=====================\nTotal Rg average is: {0:2.3f} +/- {1:2.5f}'.format(RgAvg,RgErr)
    lines += '\nTotal Rg avg. correlation time: {0:5.4f} +/- {1:5.6f}'.format(RgCorrTime, RgCorrTimeErr)
    lines += '\n\nTotal Ree average is: {0:2.3f} +/- {1:2.5f}'.format(ReeAvg,ReeErr)
    lines += '\nTotal Ree avg. correlation time: {0:5.4f} +/- {1:5.6f}'.format(ReeCorrTime, ReeCorrTimeErr)

    print(lines)
    txtRg += lines
    f = open(RgStatOutName+Ext,'w')
    f.write(txtRg)
    return  RgAvg,RgStd,RgErr,RgCorrTime,RgCorrTimeErr,RgNUncorrSamples, ReeAvg,ReeStd,ReeErr,ReeCorrTime,ReeCorrTimeErr,ReeNUncorrSamples 

def getStats(trajFile, top, NP, ThermoLog, DOP = 10, NAtomsPerChain = None,  
             StatsFName = 'AllStats.dat', RgDatName = 'RgTimeSeries', ReeDatName = 'ReeTimeSeries',RgStatOutName = 'RgReeStats', Ext='.dat',  
             fi = 'lammps', obs = None, cols = None,
             res0Id = 0, stride = 1, autowarmup = True, warmup = 100, plot = False):
    
    txt = '#  Avg.\tS.D.\tStdErr.\tCorr.\tStdErr.\tUncorr.Samples\n'
    if NP > 0:
        RgAvg,RgStd,RgErr,RgCorrTime,RgCorrTimeErr,RgNUncorrSamples, ReeAvg,ReeStd,ReeErr,ReeCorrTime,ReeCorrTimeErr,ReeNUncorrSamples = getRgRee(trajFile, top, DOP, NP, NAtomsPerChain = NAtomsPerChain,
             RgDatName = RgDatName, ReeDatName = ReeDatName, RgStatOutName = RgStatOutName, Ext=Ext,
             res0Id = res0Id, stride = stride, autowarmup = autowarmup, warmup = warmup, plot = plot)
        txt += ' Rg\t%8.5f\t%8.5f\t%8.5f\t%8.5f\t%8.5f\t%i' %(RgAvg,RgStd,RgErr,RgCorrTime,RgCorrTimeErr,RgNUncorrSamples)
        txt += '\n Ree\t%8.5f\t%8.5f\t%8.5f\t%8.5f\t%8.5f\t%i' %(ReeAvg,ReeStd,ReeErr,ReeCorrTime,ReeCorrTimeErr,ReeNUncorrSamples)

    print('reading thermo file {}'.format(ThermoLog))
    obsID, Stats = getThermo(ThermoLog, fi = fi, obs = obs, cols = cols, autowarmup = autowarmup, warmup = warmup)
    
    for i, obs in enumerate(obsID):
        Avg,Std,CorrTime,Err,NUncorrSamples = Stats[i]
        try:
            txt +=  '\n %s\t%8.5f\t%8.5f\t%8.5f\t%8.5f\t%s\t%i' %(obs, Avg, Std, Err, CorrTime, 'N/A',NUncorrSamples)
        except:
            txt +=  '\n %s\t%8.5f\t%8.5f\t%8.5f\t%8.5f\t%s\t%s' %(obs, Avg, Std, Err, CorrTime, 'N/A',NUncorrSamples)
    f = open(StatsFName, 'w')
    f.write(txt)

if __name__ ==  '__main__':

    TrajFile = __TrajFile__
    ThermoLog = __ThermoLog__
    NAtomsPerChain = __NAtomsPerChain__ #None if analyzing AA
    NP = __NP__
    top = __top__     
    DOP = __DOP__
    fi = __fi__ #'lammps' or 'openmm' 
    getStats(TrajFile, top, NP, ThermoLog, DOP = DOP, NAtomsPerChain = NAtomsPerChain, StatsFName = 'AllStats.dat',
            RgDatName = 'RgTimeSeries', ReeDatName = 'ReeTimeSeries',RgStatOutName = 'RgReeStats', Ext='.dat',
             fi = fi, obs = ['PotEng', 'Temp', 'Press'], cols = None,
             res0Id = 0, stride = 1, autowarmup = True, warmup = 100, plot = True)

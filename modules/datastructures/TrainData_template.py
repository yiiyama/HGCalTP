
from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy 

class TrainData_ID(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        self.treename="clusters" #input root tree name
        
        self.truthclasses=['electron',
                           'muon',
                           'photon',
                           'pi0',
                           'neutral',
                           'charged'] #truth classes for classification
        
        self.weightbranchX='cluster_energy' #needs to be specified
        self.weightbranchY='cluster_eta' #needs to be specified
        
        self.referenceclass='electron'
        self.weight_binX = numpy.array([0,3,5,10,20,40,70,100,150,200,300,500,40000],dtype=float) 
        self.weight_binY = numpy.array([1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,3.0],dtype=float) 
        
        
        self.registerBranches(['cluster_energy','cluster_eta']) #list of branches to be used 
        
        self.registerBranches(self.truthclasses)
        
        
        #call this at the end
        self.reduceTruth(None)
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
    
        # this function defines how to convert the root ntuple to the training format
        # options are not yet described here
        
        fileTimeOut(filename,120)
        import ROOT
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        from DeepJetCore.preprocessing import read2DArray
        
        # user code
        feature_array = read2DArray(filename,self.treename,"binned_features",self.nsamples,32,32)
        truth = self.read_truthclasses(filename)
        
        notremoves=weighter.createNotRemoveIndices(Tuple)
        
        # this removes parts of the dataset for weighting the events
        if self.remove:
            feature_array = feature_array[notremoves > 0]
            truth = truth[notremoves > 0]
        # call this in the end
        
        self.nsamples=len(feature_array)
        
        self.x=[] # list of feature numpy arrays
        self.y=[] # list of target numpy arrays (truth)
        self.w=[] # list of weight arrays. One for each truth target


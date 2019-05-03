
from DeepJetCore.TrainData import TrainData
from DeepJetCore.preprocessing import MeanNormZeroPad
import numpy 

class TrainData_binned(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        self.treename="clusters" #input root tree name
        
        self.truthclasses=['electron', 'muon', 'photon', 'pi0', 'neutral', 'charged'] #truth classes for classification
        
        self.weightbranchX='pt' #needs to be specified
        self.weightbranchY='eta' #needs to be specified
        
        self.referenceclass='flatten'
        self.weight_binX = numpy.array([0,40000],dtype=float) 
        self.weight_binY = numpy.array([-40000,40000],dtype=float) 
        
        self.addBranches(['pt', 'eta']) #list of branches to be used 

        self.channels = ['bin_x_1', 'bin_y_1', 'bin_z_1', 'bin_energy_1', 'bin_x_2', 'bin_y_2', 'bin_z_2', 'bin_energy_2']

        self.addBranches(self.channels, 950)
        
        self.registerBranches(self.truthclasses)
        
        #call this at the end
        self.reduceTruth(None)
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
    
        # this function defines how to convert the root ntuple to the training format
        # options are not yet described here

        feature_array = self.readTreeFromRootToTuple(filename)

        #notremoves=weighter.createNotRemoveIndices(Tuple)
        
        # this removes parts of the dataset for weighting the events
        #feature_array = feature_array[notremoves > 0]
                
        # call this in the end
        
        self.nsamples=len(feature_array)

        x_all = MeanNormZeroPad(filename, TupleMeanStd, self.branches, self.branchcutoffs, self.nsamples)

        self.x=[x_all] # list of feature numpy arrays
        self.y=[numpy.vstack(feature_array[self.truthclasses]).transpose()] # list of target numpy arrays (truth)
        self.w=[] # list of weight arrays. One for each truth target


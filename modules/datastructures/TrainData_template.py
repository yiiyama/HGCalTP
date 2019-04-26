
from DeepJetCore.TrainData import TrainData
import numpy 

class TrainData_template(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        self.treename="tree" #input root tree name
        
        self.truthclasses=[''] #truth classes for classification
        
        self.weightbranchX='branchx' #needs to be specified
        self.weightbranchY='branchy' #needs to be specified
        
        self.referenceclass='flatten'
        self.weight_binX = numpy.array([0,-40000,40000],dtype=float) 
        self.weight_binY = numpy.array([-40000,40000],dtype=float) 
        
        
        self.registerBranches(['']) #list of branches to be used 
        
        self.registerBranches(self.truthclasses)
        
        
        #call this at the end
        self.reduceTruth(None)
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
    
        # this function defines how to convert the root ntuple to the training format
        # options are not yet described here
        
        
        # user code
        feature_array = function_to_create_the_array(filename)
        
        notremoves=weighter.createNotRemoveIndices(Tuple)
        
        # this removes parts of the dataset for weighting the events
        feature_array = feature_array[notremoves > 0]
                
        # call this in the end
        
        self.nsamples=len(feature_array)
        
        self.x=[] # list of feature numpy arrays
        self.y=[] # list of target numpy arrays (truth)
        self.w=[] # list of weight arrays. One for each truth target


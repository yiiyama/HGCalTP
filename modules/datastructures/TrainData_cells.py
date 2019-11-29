from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
import uproot

class TrainData_cells(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        self.treename="clusters" #input root tree name

        self.truthclasses=[
            'electron',
            'muon',
            'photon',
            'pi0',
            'neutral',
            'charged'
        ] #truth classes for classification

        self.weightbranchX='cluster_pt' #needs to be specified
        self.weightbranchY='cluster_eta' #needs to be specified

        self.referenceclass='electron'
        self.weight_binX = np.array([0,3,5,10,20,40,70,100,150,200,300,500,40000],dtype=float)
        self.weight_binY = np.array([1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,3.0],dtype=float)

        self.registerBranches(['cluster_pt', 'cluster_eta']) #list of branches to be used
        self.registerBranches(self.truthclasses)

        #call this at the end
        self.reducedtruthclasses = ['egamma', 'muon', 'pi0', 'hadron']

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        # this function defines how to convert the root ntuple to the training format
        # options are not yet described here

        fileTimeOut(filename,120)

        uproot_tree = uproot.open(filename)['clusters']

        cluster_pt = uproot_tree.array('cluster_pt')
        pt_filter = cluster_pt > 5.

        n_cell = uproot_tree.array('n_cell')

        def to_ndarray(*args):
            return np.stack(args, axis=-1)

        branches = [
            'cell_energy',
            'cell_theta',
            'cell_phi',
            'cell_z'
        ]

        print("reading feature array")
        feature_array = uproot_tree.arrays(branches, outputtype=to_ndarray)
        print(feature_array.shape)

        print("reading truth")
        #truth = self.read_truthclasses(filename)
        truth = uproot_tree.arrays(self.truthclasses, outputtype=to_ndarray)

        egamma = truth[..., 0:1] + truth[..., 2:3]
        muon = truth[..., 1:2]
        pi0 = truth[..., 3:4]
        hadron = truth[..., 4:5] + truth[..., 5:6]
        truth = np.concatenate((egamma, muon, pi0, hadron), axis=-1)

        print("creating remove indxs")
        Tuple = self.readTreeFromRootToTuple(filename)
        notremoves=weighter.createNotRemoveIndices(Tuple)

        notremoves += pt_filter

        # this removes parts of the dataset for weighting the events
        if self.remove:
            feature_array = feature_array[notremoves > 0]
            n_cell = n_cell[notremoves > 0]
            truth = truth[notremoves > 0]
        # call this in the end

        self.nsamples=len(feature_array)

        self.x=[feature_array, n_cell] # list of feature numpy arrays
        self.y=[truth] # list of target numpy arrays (truth)
        self.w=[] # list of weight arrays. One for each truth target

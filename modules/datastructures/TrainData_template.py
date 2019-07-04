from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy
import uproot

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

        uproot_tree = uproot.open(filename)['clusters']

        def to_ndarray(*args):
            return numpy.squeeze(numpy.dstack(args))

        branches_template = [
            'bin_eta',
            'bin_theta',
            'bin_phi',
            'bin_x',
            'bin_y',
            'bin_eta_global',
            'bin_theta_global',
            'bin_phi_global',
            'bin_dist_global',
            'bin_x_global',
            'bin_y_global',
            'bin_z_global',
            'bin_energy',
            'bin_layer'
        ]
        branches = []
        for icell in range(3):
            branches.extend([b + ('_%d' % icell) for b in branches_template])

        feature_array = uproot_tree.arrays(branches, outputtype=to_ndarray)
        feature_array = numpy.reshape(feature_array, (-1, 5, 5, 38, 42))
        #print(feature_array)

        print("reading truth")
        #truth = self.read_truthclasses(filename)
        truth = uproot_tree.arrays(self.truthclasses, outputtype=to_ndarray)
        #print(truth)
        # for binary
        #truth = numpy.argmax(truth, axis=1)

        Tuple = self.readTreeFromRootToTuple(filename)

        print("creating remove indxs")
        notremoves=weighter.createNotRemoveIndices(Tuple)

        # this removes parts of the dataset for weighting the events
        if self.remove:
            feature_array = feature_array[notremoves > 0]
            truth = truth[notremoves > 0]
        # call this in the end

        self.nsamples=len(feature_array)

        self.x=[feature_array] # list of feature numpy arrays
        self.y=[truth] # list of target numpy arrays (truth)
        self.w=[] # list of weight arrays. One for each truth target

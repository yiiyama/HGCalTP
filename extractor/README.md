Ntuples extractor
=================

This package is for making ROOT trees to be converted into DeepJetCore training format. We have a somewhat roundabout input structure:

  CMS dataset -> HGCalTriggerNtuples -> tree from this package -> DeepJetCore training format

We need the extra step because HGCalTriggerNtuples is a comprehensive ntuple format that is not easily convertible to the DeepJetCore format.

Running the extractor
=====================

For a single file,
```
./extractor.py --in input.root --out output.root
```

To extract trees from a list of HGCalTriggerNtuples files using HTCondor,
```
./extractor.py --in-list filelist --out /eos/cms/store/user/your_directory --batch microcentury
```
Output files are named by HTCondor job ids.

Extractor macros
================

As we agreed to start with classification of predefined clusters, the only extractor macros currently provided (`clusters.cc` and `clusters_binned.cc`) take the HGCalTriggerNtuples, which is an event-wise (one event per entry) tree, and make a cluster-wise (one cluster per entry) tree. The macros should be straightforward (dumb) enough to be modified according to needs.

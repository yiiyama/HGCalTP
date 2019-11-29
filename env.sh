#! /bin/bash
THISDIR=`pwd`
export HGCALTP=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$HGCALTP
export DEEPJETCORE=/afs/cern.ch/user/y/yiiyama/src/DeepJetCore

#adapt this and then remove the exit
source $DEEPJETCORE/env.sh

export PYTHONPATH=$HGCALTP/modules:$PYTHONPATH
export PYTHONPATH=$HGCALTP/modules/datastructures:$PYTHONPATH


#! /bin/bash
THISDIR=`pwd`
export HGCALTP=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$HGCALTP


#adapt this and then remove the exit
cd /afs/cern.ch/user/j/jkiesele/work/TESTDJ/DeepJetCore
source env.sh
cd $HGCALTP
export PYTHONPATH=$HGCALTP/modules:$PYTHONPATH
export PYTHONPATH=$HGCALTP/modules/datastructures:$PYTHONPATH

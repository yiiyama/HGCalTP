#! /bin/bash
THISDIR=`pwd`
export HGCALTP=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$HGCALTP


#adapt this and then remove the exit
cd /afs/cern.ch/user/y/yiiyama/src/DeepJetCore

if command -v nvidia-smi > /dev/null
then
        source gpu_env.sh
else
        source lxplus_env.sh
fi

cd $HGCALTP
export PYTHONPATH=$HGCALTP/modules:$PYTHONPATH
export PYTHONPATH=$HGCALTP/modules/datastructures:$PYTHONPATH

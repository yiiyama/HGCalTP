#!/usr/bin/env python

import sys
import os
import subprocess
import tempfile
import shutil
from argparse import ArgumentParser

MACRO = 'clusters'

thisdir = os.path.dirname(os.path.realpath(__file__))

arg_parser = ArgumentParser(description = 'Run simple jobs on condor')
arg_parser.add_argument('--in', '-i', metavar='PATH', dest='in_path', nargs='+', default=[], help='Input files.')
arg_parser.add_argument('--in-list', '-I', metavar='PATH', dest='in_list', help='File with a list of input files, one file per line.')
arg_parser.add_argument('--out', '-o', metavar='PATH', dest='out_path', required=True, help='Output path or directory.')
arg_parser.add_argument('--batch', '-b', metavar='JOBFLAVOUR', dest='job_flavour', nargs='?', const='espresso', help='CERN HTCondor job flavour.')
arg_parser.add_argument('--cmst3', '-M', action = 'store_true', dest = 'cmst3', help = 'Use CMG accounting group for HTCondor.')
arg_parser.add_argument('--files-per-job', '-f', metavar='N', dest='files_per_job', default=1, type=int, help='Number of files per job.')
arg_parser.add_argument('--num-events', '-n', metavar='N', dest='num_events', default=-1, type=int, help='Number of events to process.')
arg_parser.add_argument('--min-pt', '-p', metavar='VAL', dest='min_pt', default=0., type=float, help='Minimum cluster pT.')
arg_parser.add_argument('--no-compile', '-C', action='store_true', dest='no_compile', help='Do not compile the ROOT macro.')
arg_parser.add_argument('--script-dir', '-d', metavar='PATH', dest='script_dir', default=thisdir, help='(Internal use) Location of the script.')

args = arg_parser.parse_args()
sys.argv = []

import ROOT

in_files = list(args.in_path)
if args.in_list is not None:
    with open(args.in_list) as source:
        for line in source:
            in_files.append(line.strip())

if args.job_flavour is not None:
    if not args.no_compile:
        ROOT.gROOT.LoadMacro('%s/%s.cc+' % (args.script_dir, MACRO))
        try:
            ROOT.extractNtuples
        except AttributeError:
            print 'Failed to compile the extractor macro.'

    logdir = '%s/logs' % args.script_dir
    try:
        os.makedirs(logdir)
    except OSError:
        pass

    if args.files_per_job < 1:
        raise RuntimeError('Invalid files-per-job')

    if os.path.exists(args.out_path) and not os.path.isdir(args.out_path):
        raise RuntimeError('--out must be a directory name')

    try:
        os.makedirs(args.out_path)
    except OSError:
        pass

    if args.cmst3:
        accounting_group = 'group_u_CMST3.all'
    else:
        accounting_group = 'group_u_CMS.u_zh'

    jdl = []
    jdl.append(('executable', '%s/setenv_exec.sh' % thisdir))
    jdl.append(('universe', 'vanilla'))
    jdl.append(('should_transfer_files', 'YES'))
    jdl.append(('input', '/dev/null'))
    jdl.append(('requirements', 'Arch == "X86_64" && OpSysAndVer == "CentOS7"'))
    jdl.append(('transfer_output_files', '""'))
    jdl.append(('accounting_group', accounting_group))
    jdl.append(('+AccountingGroup', accounting_group))
    jdl.append(('+JobFlavour', '"%s"' % args.job_flavour))
    jdl.append(('on_exit_hold', '(ExitBySignal == True) || (ExitCode != 0)'))
    jdl.append(('log', '%s/$(Cluster).$(Process).log' % logdir))
    jdl.append(('output', '%s/$(Cluster).$(Process).out' % logdir))
    jdl.append(('error', '%s/$(Cluster).$(Process).err' % logdir))
    jdl.append(('arguments', '%s --no-compile --in $(InFile) --out %s/$(Cluster)_$(Process).root --min-pt %f --script-dir %s' % (os.path.realpath(__file__), args.out_path, args.min_pt, args.script_dir)))

    jdl_text = ''.join(['%s = %s\n' % (key, str(value)) for key, value in jdl])
    jdl_text += 'queue 1 InFile from (\n'

    for i in range(0, len(in_files), args.files_per_job):
        jdl_text += ' '.join(in_files[i:i + args.files_per_job]) + '\n'

    jdl_text += ')\n'

    proc = subprocess.Popen(['condor_submit'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate(jdl_text)
    print out.strip()

else:
    if os.path.isdir(args.out_path):
        raise RuntimeError('--out must be a file name')

    if args.no_compile:
        ROOT.gSystem.Load('%s/%s_cc.so' % (args.script_dir, MACRO))
    else:
        ROOT.gROOT.LoadMacro('%s/%s.cc+' % (args.script_dir, MACRO))

    tree = ROOT.TChain('hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    for path in in_files:
        tree.Add(path)

    tmp = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
    tmp.close()

    ROOT.extractNtuples(tree, tmp.name, args.min_pt, args.num_events)

    shutil.copyfile(tmp.name, args.out_path)
    os.unlink(tmp.name)

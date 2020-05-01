#!/usr/bin/python

# parses binary output from MC-MP2 with DIMER_PRINT enabled
# MP2CV = 2, i.e. 6 control variates

import numpy as np
import sys
import json
import glob
import re
from argparse import ArgumentParser

import control_variates as CV


def print_header():
    header_format = "{:10}  " + "{:16}  " * 4
    header = header_format.format("Step", "Avg. E", "Err. E", "Avg. E Ctrled", "Err. E Ctrled")
    print(header)


def main(args):

    emp2cv = lambda s: s.rstrip(".emp.bin") + ".cv.bin"

    if args.single:
        emp_filename = args.single
        cv_filename = emp2cv(emp_filename)

        print_header()
        print(CV.from_trajectory_file(emp_filename, cv_filename))
        return 0

    if args.auto_dimer:
        # TODO: redo the code here
        dmm_jobnames = list(range(3))
        for filename in glob.glob("*.bin"):
            first_word = filename.split(".")[0]
            jobname = filename.split(".taskid_")[0]
            if first_word.endswith("dimer"):
                dmm_jobnames[0] = jobname
            elif first_word.endswith("monomer_a"):
                dmm_jobnames[1] = jobname
            elif first_word.endswith("monomer_b"):
                dmm_jobnames[2] = jobname

        print("Detected dimer jobname:     ", dmm_jobnames[0])
        print("Detected monomer A jobname: ", dmm_jobnames[1])
        print("Detected monomer B jobname: ", dmm_jobnames[2])
        if input("Is this correct? [Y/n]: ") in ["n", "N"]:
            print("Aborted per user direction.")
            return -1
    else:
        dmm_jobnames = args.dimer


    dimer_jobname = dmm_jobnames[0]
    data = CV.CV()
    for dimer_emp_file in glob.glob(dimer_jobname + "*" + args.extension + ".emp.bin"):
        monomer_a_emp_file = re.sub("dimer", "monomer_a", dimer_emp_file)
        monomer_b_emp_file = re.sub("dimer", "monomer_b", dimer_emp_file)

        dimer_cv_file = emp2cv(dimer_emp_file)
        monomer_a_cv_file = emp2cv(monomer_a_emp_file)
        monomer_b_cv_file = emp2cv(monomer_b_emp_file)

        emp = np.fromfile(dimer_emp_file) - np.fromfile(monomer_a_emp_file) - np.fromfile(monomer_b_emp_file)
        cv = np.fromfile(dimer_cv_file) - np.fromfile(monomer_a_cv_file) - np.fromfile(monomer_b_cv_file)
        cv = cv.reshape(emp.size, -1)

        data = data + CV.from_trajectory(emp, cv)

    # include json files of previous calculations, if any
    if args.include_json:
        for json_filename in args.include_json:
            data = data + CV.from_json(json_filename)

    # print CV object
    print_header()
    print(data)

    # write summary to JSON file
    json_filename = "DIMER_ANALYSIS_" + dimer_jobname.replace("_dimer", "") + ".{0}.json".format(args.extension)
    data.to_json(json_filename)

    # TODO: implement args.debug


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action = "store_true",
                        help = "run in debug mode and print calculated trajectory every 128 steps per thread. significantly slows script. (currently not implemented)")
    parser.add_argument('-e', '--extension', default = '22', help='sets extension of bin files.')
    parser.add_argument('--include-json', nargs = '*', metavar = '[JSON FILES]',
                        help = "specify JSON trajectories of previous calculations to include in energy calculation.")

    dimer_calc = parser.add_mutually_exclusive_group()
    dimer_calc.add_argument("--single", metavar = "[EMP FILE NAME]",
                            help = "process energy and control variate trajectory from a single file.")
    dimer_calc.add_argument("--dimer", nargs = 3, metavar=("[DIMER NAME]", "[MONOMER A NAME]", "[MONOMER B NAME]"),
                            help = "run in dimer mode and calculate stabilization energy. requires specifying dimer and both monomer job names (without .taskid_[n].[cv,emp].bin suffix).")
    dimer_calc.add_argument("--auto-dimer", action="store_true",
                            help = "same as '--dimer', but automatically finds job names by searching for files in the current directory containing 'dimer', 'monomer_a', 'monomer_b'.")

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        main(args)

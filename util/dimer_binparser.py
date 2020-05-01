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
        try:
            dimer_emp_file_0 = glob.glob("*dimer*.taskid_0." + args.extension + ".emp.bin")[0]
            dimer_jobname = dimer_emp_file_0.split(".taskid_")[0]
        except IndexError:
            print("Failed to find dimer file with '--auto-dimer'. Please manually specify it with '--dimer'.")
            return -1

        print("Detected dimer jobname:     ", dimer_jobname)
        print("Detected monomer A jobname: ", re.sub("dimer", "monomer_a", dimer_jobname))
        print("Detected monomer B jobname: ", re.sub("dimer", "monomer_b", dimer_jobname))
        if input("Is this correct? [Y/n]: ") in ["n", "N"]:
            print("Aborted per user direction.")
            return -1
    else:
        dimer_jobname = args.dimer

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
    dimer_calc.add_argument("--single", metavar = "[EMP FILENAME]",
                            help = "process energy and control variate trajectory from a single file.")
    dimer_calc.add_argument("--dimer", nargs = 1, metavar="[DIMER EMP FILENAME]",
                            help = "run in dimer mode and calculate stabilization energy. requires specifying dimer emp filename.")
    dimer_calc.add_argument("--auto-dimer", action="store_true",
                            help = "same as '--dimer', but takes no arguments and automatically finds dimer filename by searching for files in the current directory containing 'dimer'.")

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        main(args)

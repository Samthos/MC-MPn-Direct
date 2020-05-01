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
    if(args.single):
        # TODO
        file_dict, jobname = load_data(args.single, args.extension)
    else:
        if (args.auto_dimer):
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
    new_data = CV.CV()
    for dimer_emp_file in glob.glob(dimer_jobname + "*" + args.extension + ".emp.bin"):
        monomer_a_emp_file = re.sub("dimer", "monomer_a", dimer_emp_file)
        monomer_b_emp_file = re.sub("dimer", "monomer_b", dimer_emp_file)

        emp2cv = lambda s: s.rstrip(".emp.bin") + ".cv.bin"
        dimer_cv_file = emp2cv(dimer_emp_file)
        monomer_a_cv_file = emp2cv(monomer_a_emp_file)
        monomer_b_cv_file = emp2cv(monomer_b_emp_file)

        emp = np.fromfile(dimer_emp_file) - np.fromfile(monomer_a_emp_file) - np.fromfile(monomer_b_emp_file)
        cv = np.fromfile(dimer_cv_file) - np.fromfile(monomer_a_cv_file) - np.fromfile(monomer_b_cv_file)
        cv = cv.reshape(emp.size, -1)

        new_data = new_data + CV.from_trajectory(emp, cv)

    print_header()
    print(new_data)

    # TODO: specify json files to average over
    # old_data = sum([CV.CV.from_json(json_file) for json_file in glob.glob("*.json")])

    json_filename = "DIMER_ANALYSIS_" + dimer_jobname.replace("_dimer", "") + ".{0}.json".format(args.extension)
    data = new_data
    data.to_json(json_filename)


#        dimer_file_dict, dimer_jobname = load_data(dmm_jobnames[0], args.extension)
#        monomerA_file_dict, monomerA_jobname = load_data(dmm_jobnames[1], args.extension)
#        monomerB_file_dict, monomerB_jobname = load_data(dmm_jobnames[2], args.extension)
#
#        file_dict = dict()
#
#        jobname = "DIMER_ANALYSIS_" + dimer_jobname.replace("_dimer", "")
#        for i in dimer_file_dict:
#            file_dict[i] = [dimer_file_dict[i][0] - monomerA_file_dict[i][0] - monomerB_file_dict[i][0],
#                            dimer_file_dict[i][1] - monomerA_file_dict[i][1] - monomerB_file_dict[i][1]]
#
#
#    if args.debug:
#        for taskid in file_dict:
#            cv_data = file_dict[taskid][0]
#            emp_data = file_dict[taskid][1]
#            print("TASKID: ", taskid)
#            print("Step \t Avg. E \t Err. E \t Avg. E Ctrled \t Err. E Ctrled")
#            for step in range(128, emp_data.size + 1, 128):
#                json_filename =  None
#                if step == emp_data.size:
#                    json_filename = jobname + ".taskid_" + str(taskid) + ".22.json"
#                analysis = control_variate_analysis(emp_data[:step], cv_data[:step], json_filename)
#                print("{} \t {:.7f} \t {:.7f} \t {:.7f} \t {:.7f}".format(step, *analysis))
#    else:
#        cv_data = np.concatenate([file_dict[taskid][0] for taskid in file_dict], axis = 0)
#        emp_data = np.concatenate([file_dict[taskid][1] for taskid in file_dict], axis = 0)
#        json_filename = jobname + ".22.json"
#        step = emp_data.size
#        analysis = control_variate_analysis(emp_data, cv_data, json_filename)
#        row_header_format = "{:10}  " + "{:16}  " * 4
#        row_format = "{:<10d}  " + "{:<+16.8e}  " * 4
#        print(row_header_format.format("Step", "Avg. E", "Err. E", "Avg. E Ctrled", "Err. E Ctrled"))
#        print(row_format.format(step, *analysis))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action = "store_true",
                        help = "run in debug mode and print calculated trajectory every 128 steps per thread. significantly slows script.")
    parser.add_argument('-e', '--extension', default='22', help='sets extension of bin files.')
    dimer_calc = parser.add_mutually_exclusive_group()
    dimer_calc.add_argument("--single", metavar = "[JOB NAME]",
                            help = "extract energy and control variate data from a single job (req. job name without .taskid_[n].[cv,emp].bin suffix).")
    dimer_calc.add_argument("--dimer", nargs = 3, metavar=("[DIMER NAME]", "[MONOMER A NAME]", "[MONOMER B NAME]"),
                            help = "run in dimer mode and calculate stabilization energy. requires specifying dimer and both monomer job names (without .taskid_[n].[cv,emp].bin suffix).")
    dimer_calc.add_argument("--auto-dimer", action="store_true",
                            help = "same as '--dimer', but automatically finds job names by searching for files in the current directory containing 'dimer', 'monomer_a', 'monomer_b'.")

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        main(args)

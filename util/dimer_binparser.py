#!/usr/bin/python

# parses binary output from MC-MP2 with DIMER_PRINT enabled
# MP2CV = 2, i.e. 6 control variates

import numpy as np
import os
import os.path
import sys
import json
from argparse import ArgumentParser


def load_data(jobname):
    # force paths to be relative to cwd e.g. removes initial './'
    jobname = os.path.relpath(jobname)
    filenames = os.scandir('.')
    cv_files = []
    emp_files = []
    taskids = set()

    for filename in filenames:
        filename = filename.name
        if(filename.startswith(jobname) and os.path.isfile(filename)):
            taskid = get_taskid(filename)
            if taskid != -1:
                taskids.add(taskid)

            if filename.endswith(".cv.bin"):
                cv_files.append(filename)
            elif filename.endswith(".emp.bin"):
                emp_files.append(filename)

    taskids = sorted(taskids)
    file_list = zip(sorted(cv_files), sorted(emp_files))
    file_list = [[np.fromfile(pair[0]).reshape(-1, 6), np.fromfile(pair[1])] for pair in file_list]
    file_dict = dict(zip(taskids, file_list))

    return file_dict, jobname


def get_taskid(filename):
    taskid = -1
    for s in filename.split('.'):
        if s.startswith("taskid_"):
            taskid = int(s[7:])

    return taskid


def control_variate_analysis(emp, cv, json_filename=None):
    step = emp.size

    E_avg = np.average(emp)
    E_var = np.var(emp)
    E_err = np.sqrt(E_var / step)

    cv_avg = np.average(cv, axis=0)
    E_cv_cov = np.cov(cv, emp[:step], rowvar=False, bias=True)[-1, :6]
    cv_cv_cov = np.cov(cv, rowvar=False, bias=True)
    alpha = np.linalg.solve(cv_cv_cov, E_cv_cov)

    E_avg_ctrl = E_avg - np.dot(alpha, cv_avg)
    E_var_ctrl = E_var - np.dot(alpha, E_cv_cov)
    E_err_ctrl = np.sqrt(E_var_ctrl / step) if E_var_ctrl >= 0 else -1

    if json_filename is not None:
        EX = E_avg
        EX2 = np.average(emp ** 2)
        EC = cv_avg
        EXC = np.average(np.multiply(cv.T, emp), axis = 1)
        COVXC = EXC - EC * EX
        ECC = (1 / step) * np.einsum('ij,ik->jk', cv, cv)
        COVCC = ECC - np.outer(EC, EC)
        json_data = {'STEPS': step,
                     'EX': EX,
                     'EX2': EX2,
                     'EC': EC.tolist(),
                     'EXC': EXC.tolist(),
                     'COVXC': COVXC.tolist(),
                     'alpha': alpha.tolist(),
                     'ECC': ECC.tolist(),
                     'COVCC': COVCC.tolist()}
        to_json(json_data, json_filename)
    return E_avg, E_err, E_avg_ctrl, E_err_ctrl


def to_json(output, json_filename):
    with open(json_filename, mode = 'w', encoding = 'utf-8') as json_file:
        json.dump(output, json_file, separators = (',', ':'), indent = 4)


def main(args):
    if(args.single):
        file_dict, jobname = load_data(args.single)
    elif(args.auto_dimer):
        # TODO
        # find dimer files automatically
        pass
    else:
        # i.e. dimer and monomers are explicitly specified in args.dimer
        dimer_file_dict, dimer_jobname = load_data(args.dimer[0])
        monomerA_file_dict, monomerA_jobname = load_data(args.dimer[1])
        monomberB_file_dict, monomerB_jobname = load_data(args.dimer[2])

        file_dict = dict()
        jobname = "DIMER_ANALYSIS_" + dimer_jobname
        for i in dimer_file_dict:
            file_dict[i][0] = dimer_file_dict[i][0] - monomerA_file_dict[i][0] - monomerB_file_dict[i][0]
            file_dict[i][1] = dimer_file_dict[i][1] - monomerA_file_dict[i][1] - monomerB_file_dict[i][1]

    if args.debug:
        for taskid in file_dict:
            cv_data = file_dict[taskid][0]
            emp_data = file_dict[taskid][1]
            print("TASKID: ", taskid)
            print("Step \t Avg. E \t Err. E \t Avg. E Ctrled \t Err. E Ctrled")
            for step in range(128, emp_data.size + 1, 128):
                json_filename =  None
                if step == emp_data.size:
                    json_filename = jobname + ".taskid_" + str(taskid) + ".22.json"
                analysis = control_variate_analysis(emp_data[:step], cv_data[:step], json_filename)
                print("{} \t {:.7f} \t {:.7f} \t {:.7f} \t {:.7f}".format(step, *analysis))
    else:
        cv_data = np.concatenate([file_dict[taskid][0] for taskid in file_dict], axis = 0)
        emp_data = np.concatenate([file_dict[taskid][1] for taskid in file_dict], axis = 0)
        json_filename = jobname + ".22.json"
        step = emp_data.size
        analysis = control_variate_analysis(emp_data, cv_data, json_filename)
        print("Step \t Avg. E \t Err. E \t Avg. E Ctrled \t Err. E Ctrled")
        print("{} \t {:.7f} \t {:.7f} \t {:.7f} \t {:.7f}".format(step, *analysis))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug",
                        help = "run in debug mode and print calculated trajectory every 128 steps per thread. significantly slows script.",
                        action = "store_true")
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

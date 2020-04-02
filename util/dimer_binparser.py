#!/usr/bin/python

# parses binary output from MC-MP2 with DIMER_PRINT enabled
# MP2CV = 2, i.e. 6 control variates

import numpy as np
import os
import os.path
import sys

def load_data(jobname):
    # force paths to be relative to cwd e.g. removes initial './'
    jobname = os.path.relpath(jobname)
    files = os.scandir('.')
    cv_files = []
    emp_files = []
    taskids = set()

    for file in files:
        filename = file.name
        if(filename.startswith(jobname) and os.path.isfile(filename)):
            taskid = get_taskid(filename)
            if taskid != -1: taskids.add(taskid)
            if filename.endswith(".cv.bin"): cv_files.append(filename)
            if filename.endswith(".emp.bin"): emp_files.append(filename)

    taskids = sorted(taskids)
    file_list = [list(pair) for pair in zip(sorted(cv_files), sorted(emp_files))]
    file_list = [[np.fromfile(pair[0]).reshape(-1, 6), np.fromfile(pair[1])] for pair in file_list]
    file_dict = dict(zip(taskids, file_list))

    return file_dict

def get_taskid(filename):
    taskid_index = filename.find("taskid")
    if taskid_index < 0:
        return -1
    else:
        return int(filename[taskid_index + 7])

def control_variate_analysis(emp, cv):
    step = emp.size

    E_avg = np.average(emp)
    E_var = np.var(emp)
    E_err = np.sqrt(E_var / step)

    # 6 x 1 vector of the average of all (6) control variates thus far
    cv_avg = np.average(cv, axis=0)

    E_cv_cov = np.cov(cv, emp[:step], rowvar=False, bias=True)[-1, :6]
    cv_cv_cov = np.cov(cv, rowvar=False, bias=True)
    alpha = np.linalg.solve(cv_cv_cov, E_cv_cov)

    E_avg_ctrl = E_avg - np.dot(alpha, cv_avg)
    E_var_ctrl = E_var - np.dot(alpha, E_cv_cov)
    E_err_ctrl = np.sqrt(E_var_ctrl / step) if E_var_ctrl >= 0 else -1

    return E_avg, E_err, E_avg_ctrl, E_err_ctrl


def main():
    file_dict = load_data(sys.argv[1])

    for taskid in file_dict:
        cv_data = file_dict[taskid][0]
        emp_data = file_dict[taskid][1]
        print("TASKID: ", taskid)
        print("Step \t Avg. E \t Err. E \t Avg. E Ctrled \t Err. E Ctrled")
        for step in range(128, emp_data.size + 1, 128):
            print("{} \t {:.7f} \t {:.7f} \t {:.7f} \t {:.7f}".format(step,
                *control_variate_analysis(emp_data[:step], cv_data[:step])))


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("USAGE: dimer_binparser.py <job name>")
    else:
        main()

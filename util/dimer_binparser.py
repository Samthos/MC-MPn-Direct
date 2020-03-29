#!/usr/bin/python

# parses binary output from MC-MP2 with DIMER_PRINT enabled
# MP2CV = 2, i.e. 6 control variates

import numpy as np
import re
import sys


def load_data(fname):
    emp_data = np.fromfile(fname)
    cv_data = np.fromfile(re.sub('emp.bin', 'cv.bin', fname)).reshape(-1, 6)
    return emp_data, cv_data


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
    emp_data, cv_data = load_data(sys.argv[1])

    print("Step \t Avg. E \t Err. E \t Avg. E Ctrled \t Err. E Ctrled")
    for step in range(128, emp_data.size + 1, 128):
        print("{} \t {:.7f} \t {:.7f} \t {:.7f} \t {:.7f}".format(step,
            *control_variate_analysis(emp_data[:step], cv_data[:step])))


if __name__ == main():
    if len(sys.argv) != 3:
        print("USAGE: dimer_binparser.py <emp binary file>")
    else:
        main()

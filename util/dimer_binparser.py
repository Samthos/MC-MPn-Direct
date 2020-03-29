#!/usr/bin/python

# parses binary output from MC-MP2 with DIMER_PRINT enabled
# MP2CV = 2, i.e. 6 control variates

import numpy as np
import sys

try:
    emp_filename = sys.argv[1]
    cv_filename = sys.argv[2]
except:
    print("USAGE: dimer_binparser.py [emp binary file] [cv binary file]")
    sys.exit()

emp = np.fromfile(emp_filename)
cv = np.fromfile(cv_filename)
cv = np.reshape(cv, (cv.size // 6, 6))

steps = emp.size
E_cv_cov = np.zeros(6)

# print output every 128 steps, i.e. step 128, 256, ...

print("Step \t Avg. E \t Err. E \t Avg. E Ctrled \t Err. E Ctrled")
for i in range(127, steps, 128):
    step = i + 1
    E_avg = np.average(emp[0:i + 1])
    var_E = np.var(emp[0:i + 1])
    err_E = np.sqrt(var_E / step)

    # 6 x 1 vector of the average of all (6) control variates thus far
    cv_avg = np.average(cv[0:i + 1], axis = 0)

    E_cv_cov = np.cov(cv[0:i + 1], emp[0:i + 1], rowvar = False)[6][0:6]
    cv_cv_cov = np.cov(cv, rowvar= False)
    alpha = np.linalg.solve(cv_cv_cov, E_cv_cov)

    E_avg_ctrl = E_avg - np.dot(alpha, cv_avg)
    var_E_ctrl = var_E - np.dot(alpha, E_cv_cov)
    err_E_ctrl = np.sqrt(var_E_ctrl / step) if var_E_ctrl >= 0 else -1

    print(F"{step} \t {E_avg:.7f} \t {err_E:.7f} \t {E_avg_ctrl:.7f} \t {err_E_ctrl:.7f}")

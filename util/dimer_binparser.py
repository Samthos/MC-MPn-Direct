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

emp_array = np.fromfile(emp_filename)
cv_array = np.fromfile(cv_filename)
cv_array = np.reshape(cv_array, (cv_array.size // 6, 6))

print(emp_array)
print(cv_array)

steps = emp_array.size

# print output every 128 steps
print("Step \t E \t\t Err. E")
for i in range(127, steps, 128):
    step = i + 1
    E = np.average(emp_array[0:i + 1])
    err_E = np.sqrt(np.var(emp_array[0:i + 1]) / (i + 1))

    print(F"{step} \t {E:.7f} \t {err_E:.7f}")

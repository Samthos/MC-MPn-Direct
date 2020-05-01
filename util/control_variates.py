#!/usr/bin/python
#
# defines CV class storing all energy and control variate data

import numpy as np
import json


def from_trajectory(emp, cv):
    CV_obj = CV()
    N = emp.size
    CV_obj.steps = N

    # average of energies
    CV_obj.EX = np.average(emp)

    # average of squared energies (to calculate variance)
    CV_obj.EX2 = np.average(emp ** 2)

    # vector of average control variates
    CV_obj.EC = np.average(cv, axis=0)

    # vector of average energy-control variate products
    CV_obj.EXC = np.average(np.multiply(cv.T, emp), axis = 1)

    # matrix of average control variate products
    CV_obj.ECC = (1 / N) * np.einsum('ij,ik->jk', cv, cv)

    # compute energy/control variate covariance vector
    CV_obj.COVXC = CV_obj.EXC - CV_obj.EC * CV_obj.EX

    # compute covariance matrix of control variates
    CV_obj.COVCC = CV_obj.ECC - np.outer(CV_obj.EC, CV_obj.EC)

    # compute alpha
    CV_obj.alpha = np.linalg.solve(CV_obj.COVCC, CV_obj.COVXC)

    return CV_obj


def from_trajectory_file(emp_filename):
    emp = np.fromfile(emp_filename)
    cv = np.fromfile(cv_filename).reshape(emp.size, -1)

    return from_trajectory(emp, cv)


def from_json(json_filename):
    with open(json_filename, 'r') as f:
        data = f.read()

    json_dict = json.loads(data)

    CV_obj = CV()
    CV_obj.steps = json_dict['STEPS']
    CV_obj.EX = json_dict['EX']
    CV_obj.EX2 = json_dict['EX2']
    CV_obj.EC = np.array(json_dict['EC'])
    CV_obj.EXC = np.array(json_dict['EXC'])
    CV_obj.ECC = np.array(json_dict['ECC'])
    CV_obj.COVXC = np.array(json_dict['COVXC'])
    CV_obj.alpha = np.array(json_dict['alpha'])
    CV_obj.COVCC = np.array(json_dict['COVCC'])

    return CV_obj



class CV:
    def __init__(self):
        # given/averaged quantities
        self.steps = 0
        self.EX = 0
        self.EX2 = 0
        self.EC = 0
        self.EXC = 0
        self.ECC = 0

        # computed quantities
        self.COVXC = 0
        self.alpha = 0
        self.COVCC = 0


    def __repr__(self):
        row_format = "{:<10d}  " + "{:<+16.8e}  " * 4
        row = row_format.format(self.steps, *self.analysis())
        return row


    def __radd__(self, other):
        if type(other) == int:
            return self
        else:
            message = "unsupported operand type(s) for +: '{}' and 'CV'".format(other.__class__.__name__)
            raise TypeError(message)


    def __add__(self, other):
        CV_obj = CV()
        N = self.steps + other.steps
        self_ratio = self.steps / N
        other_ratio = other.steps / N
        CV_obj.steps = N

        # averaged quantities
        CV_obj.EX = self_ratio * self.EX + other_ratio * other.EX
        CV_obj.EX2 = self_ratio * self.EX2 + other_ratio * other.EX2
        CV_obj.EC = self_ratio * self.EC + other_ratio * other.EC
        CV_obj.EXC = self_ratio * self.EXC + other_ratio * other.EXC
        CV_obj.ECC = self_ratio * self.ECC + other_ratio * other.ECC

        # re-compute energy/control variate covariance
        CV_obj.COVXC = CV_obj.EXC - CV_obj.EC * CV_obj.EX

        # re-compute covariance matrix of control variates
        CV_obj.COVCC = CV_obj.ECC - np.outer(CV_obj.EC, CV_obj.EC)

        # re-compute alpha
        CV_obj.alpha = np.linalg.solve(CV_obj.COVCC, CV_obj.COVXC)

        return CV_obj


    def to_json(self, json_filename):
        json_data = {'STEPS': self.steps,
                     'EX'   : self.EX,
                     'EX2'  : self.EX2,
                     'EC'   : self.EC.tolist(),
                     'EXC'  : self.EXC.tolist(),
                     'COVXC': self.COVXC.tolist(),
                     'alpha': self.alpha.tolist(),
                     'ECC'  : self.ECC.tolist(),
                     'COVCC': self.COVCC.tolist()}

        with open(json_filename, mode = 'w', encoding = 'utf-8') as json_file:
            json.dump(json_data, json_file, separators = (',', ':'), indent = 4)


    def analysis(self):
        N = self.steps

        # compute variance and standard error of energy
        E_avg = self.EX
        E_var = self.EX2 - (self.EX ** 2)
        E_err = np.sqrt(E_var / N)

        # compute controlled energy and its variance, standard error
        E_avg_ctrl = self.EX - np.dot(self.alpha, self.EC)
        E_var_ctrl = E_var - np.dot(self.alpha, self.COVXC)
        E_err_ctrl = np.sqrt(E_var_ctrl / N) if E_var_ctrl >= 0 else -1

        return [E_avg, E_err, E_avg_ctrl, E_err_ctrl]

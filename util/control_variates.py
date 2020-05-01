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

    # average of squared energies
    CV_obj.EX2 = np.average(emp ** 2)

    # average of each control variate
    CV_obj.EC = np.average(cv, axis=0)

    # matrix of average control variate products
    CV_obj.ECC = (1 / N) * np.einsum('ij,ik->jk', cv, cv)

    # variance and standard error
    E_var = np.var(emp)
    E_err = np.sqrt(E_var / N)

    # compute energy/control variate covariance
    CV_obj.COVXC = CV_obj.EXC - CV_obj.EC * CV_obj.EX

    # compute covariance matrix of control variates
    CV_obj.COVCC = CV_obj.ECC - np.outer(CV_obj.EC, CV_obj.EC)

    # compute alpha
    CV_obj.alpha = np.linalg.solve(CV_obj.COVCC, CV_obj.COVXC)

    E_avg_ctrl = CV_obj.EX - np.dot(CV_obj.alpha, CV_obj.EC)
    E_var_ctrl = E_var - np.dot(CV_obj.alpha, CV_obj.COVXC)
    E_err_ctrl = np.sqrt(E_var_ctrl / N) if E_var_ctrl >= 0 else -1

    CV_obj.EXC = np.average(np.multiply(cv.T, emp), axis = 1)

    CV_obj.analysis = [CV_obj.EX, E_err, E_avg_ctrl, E_err_ctrl]

    return CV_obj


def from_trajectory_file(emp, cv):
    # TODO
    pass


def from_json(filename):
    # TODO
    pass


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

        # TODO get rid of this
        self.analysis = None


    def __repr__(self):
        row_format = "{:<10d}  " + "{:<+16.8e}  " * 4
        row = row_format.format(self.steps, *self.analysis)
        return row


    def __radd__(self, other):
        return self


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

        # re-computed quantities
        # TODO: move analysis to a separate function; avoid cut and paste code

        # compute variance and standard error
        E_var = CV_obj.EX2 - CV_obj.EX
        E_err = np.sqrt(E_var / N)

        # re-compute energy/control variate covariance
        CV_obj.COVXC = CV_obj.EXC - CV_obj.EC * CV_obj.EX

        # re-compute covariance matrix of control variates
        CV_obj.COVCC = CV_obj.ECC - np.outer(CV_obj.EC, CV_obj.EC)

        # re-compute alpha
        CV_obj.alpha = np.linalg.solve(CV_obj.COVCC, CV_obj.COVXC)

        E_avg_ctrl = CV_obj.EX - np.dot(CV_obj.alpha, CV_obj.EC)
        E_var_ctrl = E_var - np.dot(CV_obj.alpha, CV_obj.COVXC)
        E_err_ctrl = np.sqrt(E_var_ctrl / N) if E_var_ctrl >= 0 else -1

        CV_obj.analysis = [CV_obj.EX, E_err, E_avg_ctrl, E_err_ctrl]
        analysis = [CV_obj.EX, E_err, E_avg_ctrl, E_err_ctrl]

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

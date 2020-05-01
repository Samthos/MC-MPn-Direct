#!/usr/bin/python
#
# defines CV class storing all energy and control variate data

import numpy as np


def from_trajectory(emp, cv):
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

    CV_obj = CV()
    CV_obj.steps = step
    CV_obj.EX = E_avg
    CV_obj.EX2 = np.average(emp ** 2)
    CV_obj.EC = cv_avg
    CV_obj.EXC = np.average(np.multiply(cv.T, emp), axis = 1)
    CV_obj.COVXC = CV_obj.EXC - CV_obj.EC * CV_obj.EX
    CV_obj.alpha = alpha
    CV_obj.ECC = (1 / step) * np.einsum('ij,ik->jk', cv, cv)
    CV_obj.COVCC = CV_obj.ECC - np.outer(CV_obj.EC, CV_obj.EC)
    CV_obj.analysis = [E_avg, E_err, E_avg_ctrl, E_err_ctrl]

    return CV_obj


def from_trajectory_file(emp, cv):
    # TODO
    pass


def from_json(filename):
    # TODO
    pass


class CV:
    def __init__(self):
        self.steps = 0
        self.EX = 0
        self.EX2 = 0
        self.EC = 0
        self.EXC = 0
        self.COVXC = 0
        self.alpha = 0
        self.ECC = 0
        self.COVCC = 0
        self.analysis = None


    def __repr__(self):
        row_format = "{:<10d}  " + "{:<+16.8e}  " * 4
        row_header_format = "{:10}  " + "{:16}  " * 4

        row_header = row_header_format.format("Step", "Avg. E", "Err. E", "Avg. E Ctrled", "Err. E Ctrled") + "\n"
        row = row_format.format(self.steps, *self.analysis)

        return row_header + row


    def __radd__(self, other):
        print("you probably shouldn't be adding a non-CV object to a CV object")
        return self


    def __add__(self, other):
        CV_obj = CV()
        N = self.steps + other.steps
        self_ratio = self.steps / N
        other_ratio = other.steps / N

        CV_obj.steps = N
        CV_obj.EX = self_ratio * self.EX + other_ratio * other.EX
        CV_obj.EX2 = self_ratio * self.EX2 + other_ratio * other.EX2
        CV_obj.EC = self_ratio * self.EC + other_ratio * other.EX2
        CV_obj.EXC = self_ratio * self.EXC + other_ratio * other.EXC
        CV_obj.COVXC = self_ratio * self.COVXC + other_ratio * other.COVXC
        CV_obj.alpha = self_ratio * self.alpha + other_ratio * other.alpha
        CV_obj.ECC = self_ratio * self.ECC + other_ratio * other.ECC
        CV_obj.COVCC = self_ratio * self.COVCC + other_ratio * other.COVCC

        if other.analysis is None:
            CV_obj.analysis = self.analysis
        elif self.analysis is None:
            CV_obj.analysis = other.analysis
        else:
            CV_obj.analysis = list(range(4))
            # energy
            CV_obj.analysis[0] = CV_obj.EX
            # std. dev. in energy
            CV_obj.analysis[1] = np.sqrt((self_ratio * self.analysis[1]) ** 2 + (other_ratio * other.analysis[1]) ** 2)
            # controlled energy
            CV_obj.analysis[2] = self_ratio * self.analysis[2] + other_ratio * other.analysis[2]
            # std. dev. in controlled energy
            CV_obj.analysis[3] = np.sqrt((self_ratio * self.analysis[3]) ** 2 + (other_ratio * other.analysis[3]) ** 2)

        return CV_obj


    def to_json(self, filename):
        # TODO
        pass

#include <iostream>
#include <cmath>

#include "shell.h"

SHELL::Shell_Type SHELL::string_to_shell_type(const std::string& str) {
  if (str == "SP") {
    return SHELL::SP;
  } else if (str == "S") {
    return SHELL::S;
  } else if (str == "P") {
    return SHELL::P;
  } else if (str == "D") {
    return SHELL::D;
  } else if (str == "F") {
    return SHELL::F;
  } else if (str == "G") {
    return SHELL::G;
  } else if (str == "H") {
    return SHELL::H;
  } else if (str == "I") {
    return SHELL::I;
  } else {
    throw Shell_Exception("When trying to read basis encountered usuported angular momentum of " + str);
  }
}
int SHELL::number_of_polynomials(SHELL::Shell_Type shell_type, bool spherical) {
  if (spherical) {
    return SHELL::number_of_spherical_polynomials(shell_type);
  } else {
    return SHELL::number_of_cartesian_polynomials(shell_type);
  }
}
int SHELL::number_of_polynomials(int shell_type, bool spherical) {
  return number_of_polynomials(static_cast<SHELL::Shell_Type>(shell_type), spherical);
}
int SHELL::number_of_spherical_polynomials(SHELL::Shell_Type shell_type) {
  int nbf = 0;
  if (shell_type == SHELL::SP) {
    nbf = 4;
  } else {
    nbf = 2 * shell_type + 1;
  }
  return nbf;
}
int SHELL::number_of_cartesian_polynomials(SHELL::Shell_Type shell_type) {
  int nbf = 0;
  if (shell_type == SHELL::SP) {
    nbf = 4;
  } else {
    nbf = (shell_type+1) * (shell_type+2) / 2;
  }
  return nbf;
}

SHELL::Shell::Shell(Shell_Type st, const Contracted_Gaussian& cg) :
  shell_type(st),
  contracted_gaussian(cg)
{
  if (shell_type != SP) {
    normalize();
  } else {
    normalize_sp();
  }
}

size_t SHELL::Shell::n_contractions() const {
  return contracted_gaussian.front().second.size();
}

void SHELL::Shell::normalize_sp() {
  for (int j = 0; j < 2; ++j) {
    Shell_Type st = S;
    if (j == 1) {
      st = P;
    }

    Contracted_Gaussian cg(contracted_gaussian.size());
    for (int i = 0; i < contracted_gaussian.size(); i++){
      cg[i].first = contracted_gaussian[i].first;
      cg[i].second.push_back(contracted_gaussian[i].second[j]);
    }

    SHELL::Shell temp_shell(st, cg);

    for (int i = 0; i < contracted_gaussian.size(); i++){
      contracted_gaussian[i].second[j] = temp_shell.contracted_gaussian[i].second[0];
    }
  }
}

double SHELL::Shell::overlap_s(double a, double b) {
  constexpr double pi32 = 5.56832799683170;
  return pi32 / pow(a + b, 1.5);
}
double SHELL::Shell::overlap_p(double a, double b) {
  constexpr double pi32 = 5.56832799683170;
  return pi32 / (2.0*pow(a + b,2.5));
}
double SHELL::Shell::overlap_d(double a, double b) {
  constexpr double pi32 = 5.56832799683170;
  return (3.0*pi32)/(4.0*pow(a + b,3.5));
}
double SHELL::Shell::overlap_f(double a, double b) {
  constexpr double pi32 = 5.56832799683170;
  return (15.0*pi32)/(8.0*pow(a + b,4.5));
}
double SHELL::Shell::overlap_g(double a, double b) {
  constexpr double pi32 = 5.56832799683170;
  return (105.0*pi32)/(16.0*pow(a + b,5.5));
}
double SHELL::Shell::overlap_general(double a, double b) {
  constexpr double pi32 = 5.56832799683170;
  int k = shell_type;
  double x = pow(a + b,-1.5 - k) * pow(2, -2 * k) * pi32;
  for (int i = k + 1; i <= 2 * k; i++) {
    x *= i;
  }
  return x;
}

double SHELL::Shell::overlap(double a, double b) {
  double norm = 0;
  switch (shell_type) {
    case S: norm = overlap_s(a, b); break;
    case P: norm = overlap_p(a, b); break;
    case D: norm = overlap_d(a, b); break;
    case F: norm = overlap_f(a, b); break;
    case G: norm = overlap_g(a, b); break;
    default: norm = overlap_general(a, b); break;
  }
  return norm;
}
void SHELL::Shell::normalize() {
  constexpr double pi = 3.141592653589793;
  constexpr double pi32 = 5.56832799683170;

  // normalize each gaussion in contraction
  for (auto & kt : contracted_gaussian) {
    double cnorm = 1 / sqrt(overlap(kt.first, kt.first));
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }

  // calculate normalization of each contraction
  std::vector<double> norm(n_contractions(), 0.00);
  for (auto first_guassian = contracted_gaussian.begin(); first_guassian != contracted_gaussian.end(); first_guassian++) {
    for (auto second_guassian = first_guassian; second_guassian != contracted_gaussian.end(); second_guassian++) {
      for (auto m = 0; m < norm.size(); m++) {
        double fac = overlap(first_guassian->first, second_guassian->first);
        double dum = first_guassian->second[m] * second_guassian->second[m] * fac;
        if (first_guassian != second_guassian) {
          dum = dum + dum;
        }
        norm[m] += dum;
      }
    }
  }
  for (auto& kt : norm) {
    kt = 1.0 / sqrt(kt);
  }

  // apply normalization to each contraction
  for (auto & kt : contracted_gaussian) {
    for (auto l = 0; l < kt.second.size(); l++) {
      kt.second[l] *= norm[l];
    }
  }
}

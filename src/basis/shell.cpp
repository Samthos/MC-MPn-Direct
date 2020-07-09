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
  } else {
    std::cerr << "When trying to read basis encountered usuported angular momentum of " << str << std::endl;
    std::exit(EXIT_FAILURE);
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
    nbf += 4;
  } else {
    nbf += 2 * shell_type + 1;
  }
  return nbf;
}
int SHELL::number_of_cartesian_polynomials(SHELL::Shell_Type shell_type) {
  int nbf = 0;
  if (shell_type == SHELL::SP) {
    nbf = 4;
  } else {
    int l = shell_type;
    nbf = (l+1) * (l+2) / 2;
  }
  return nbf;
}

SHELL::Shell::Shell(Shell_Type st, const Contracted_Gaussian& cg) :
  shell_type(st),
  contracted_gaussian(cg)
{
  switch (shell_type) {
    case SP: normalize_sp(); break;
    case S: normalize_s(); break;
    case P: normalize_p(); break;
    case D: normalize_d(); break;
    case F: normalize_f(); break;
    case G: normalize_g(); break;
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
void SHELL::Shell::normalize_s() {
  constexpr double pi = 3.141592653589793;
  constexpr double pi32 = 5.56832799683170;

  // normalize each gaussion in contraction
  for (auto & kt : contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75);
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }

  // calculate normalization of each contraction
  std::vector<double> norm(n_contractions(), 0.00);
  for (auto first_guassian = contracted_gaussian.begin(); first_guassian != contracted_gaussian.end(); first_guassian++) {
    for (auto second_guassian = first_guassian; second_guassian != contracted_gaussian.end(); second_guassian++) {
      for (auto m = 0; m < norm.size(); m++) {
        double fac = pow(first_guassian->first + second_guassian->first, 1.5);
        double dum = first_guassian->second[m] * second_guassian->second[m] / fac;
        if (first_guassian != second_guassian) {
          dum = dum + dum;
        }
        norm[m] += dum;
      }
    }
  }
  for (auto& kt : norm) {
    kt = 1.0 / sqrt(kt * pi32);
  }

  // apply normalization to each contraction
  for (auto & kt : contracted_gaussian) {
    for (auto l = 0; l < kt.second.size(); l++) {
      kt.second[l] *= norm[l];
    }
  }
}
void SHELL::Shell::normalize_p() {
  constexpr double pi = 3.141592653589793;
  constexpr double pi32 = 5.56832799683170;

  // normalize each guassian
  for (auto & kt : contracted_gaussian) {
    double cnorm = sqrt(4.0 * kt.first) * pow(2.0 * kt.first / pi, 0.75);
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }

  std::vector<double> norm(n_contractions(), 0.00);
  for (auto first_guassian = contracted_gaussian.begin(); first_guassian != contracted_gaussian.end(); first_guassian++) {
    for (auto second_guassian = first_guassian; second_guassian != contracted_gaussian.end(); second_guassian++) {
      for (auto m = 0; m < norm.size(); m++) {
        double fac = pow(first_guassian->first + second_guassian->first, 2.5);
        double dum = 0.5 * first_guassian->second[m] * second_guassian->second[m] / fac;
        if (first_guassian != second_guassian) {
          dum = dum + dum;
        }
        norm[m] += dum;
      }
    }
  }
  for (auto& kt : norm) {
    kt = 1.0 / sqrt(kt * pi32);
  }

  for (auto & kt : contracted_gaussian) {
    for (auto l = 0; l < kt.second.size(); l++) {
      kt.second[l] *= norm[l];
    }
  }
}
void SHELL::Shell::normalize_d() {
  constexpr double pi = 3.141592653589793;

  for (auto & kt : contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75) * 4.0 * kt.first / sqrt(3.0);
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }
}
void SHELL::Shell::normalize_f() {
  constexpr double pi = 3.141592653589793;

  for (auto & kt : contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75) * pow(4.0 * kt.first, 1.5) / sqrt(15.0);
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }
}
void SHELL::Shell::normalize_g() {
  constexpr double pi = 3.141592653589793;

  for (auto & kt : contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75) * pow(4.0 * kt.first,2.0) / sqrt(105.0);
    for (double &lt : kt.second) {
      lt *= cnorm;
    }
  }
}

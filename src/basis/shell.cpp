#include <iostream>

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

#ifndef SHELL_H_
#define SHELL_H_

#include <vector>
#include <string>

namespace SHELL {
  enum Shell_Type {
    SP=-1, S, P, D, F, G, H
  };
  Shell_Type string_to_shell_type(const std::string& str);
  int number_of_polynomials(Shell_Type, bool spherical);
  int number_of_polynomials(int, bool spherical);
  int number_of_spherical_polynomials(Shell_Type st);
  int number_of_cartesian_polynomials(Shell_Type st);
  struct Shell {
    Shell_Type shell_type;
    std::vector<std::pair<double, std::vector<double>>> contracted_gaussian;
    size_t n_contractions() const {
      return contracted_gaussian.front().second.size();
    }
  };
}
#endif // SHELL_H_

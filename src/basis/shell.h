#ifndef SHELL_H_
#define SHELL_H_

#include <vector>
#include <string>

namespace SHELL {
  typedef std::vector<std::pair<double, std::vector<double>>> Contracted_Gaussian;
  enum Shell_Type {
    SP=-1, S, P, D, F, G, H, NA
  };
  Shell_Type string_to_shell_type(const std::string& str);
  int number_of_polynomials(Shell_Type, bool spherical);
  int number_of_polynomials(int, bool spherical);
  int number_of_spherical_polynomials(Shell_Type st);
  int number_of_cartesian_polynomials(Shell_Type st);

  class Shell {
    public:
      Shell() = default;
      Shell(Shell_Type, const Contracted_Gaussian&);
      size_t n_contractions() const;

      Shell_Type shell_type;
      Contracted_Gaussian contracted_gaussian;

    protected:
      void normalize_sp();
      void normalize_s();
      void normalize_p();
      void normalize_d();
      void normalize_f();
      void normalize_g();
  };
}
#endif // SHELL_H_

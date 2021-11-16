#ifndef SHELL_H_
#define SHELL_H_

#include <vector>
#include <string>

namespace SHELL {
  typedef std::vector<std::pair<double, std::vector<double>>> Contracted_Gaussian;
  enum Shell_Type {
    SP=-1, S, P, D, F, G, H, I, NA
  };
  Shell_Type string_to_shell_type(const std::string& str);
  int number_of_polynomials(Shell_Type, bool spherical);
  int number_of_polynomials(int, bool spherical);
  int number_of_spherical_polynomials(Shell_Type st);
  int number_of_cartesian_polynomials(Shell_Type st);

  struct Shell_Exception : public std::exception {
    Shell_Exception(std::string str_) : str(str_) {}
    virtual const char* what() const noexcept {
      return str.c_str();
    }
    std::string str;
  };

  class Shell {
    public:
      Shell() = default;
      Shell(Shell_Type, const Contracted_Gaussian&);
      size_t n_contractions() const;

      Shell_Type shell_type;
      Contracted_Gaussian contracted_gaussian;

    protected:
      double overlap_s(double, double);
      double overlap_p(double, double);
      double overlap_d(double, double);
      double overlap_f(double, double);
      double overlap_g(double, double);
      double overlap_general(double, double);
      double overlap  (double, double);
      void normalize();

      void normalize_sp();
  };
}
#endif // SHELL_H_

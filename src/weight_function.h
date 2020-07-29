#ifndef WEIGHT_FUNCTIONS_H_
#define WEIGHT_FUNCTIONS_H_

#include <array>
#include <string>
#include <vector>

#include "molecule.h"
#include "qc_mpi.h"

struct mc_basis_typ {
  mc_basis_typ() = default;

  mc_basis_typ(const std::vector<double>& a, const std::vector<double>& n, const Point& c)
      : alpha(a), norm(n), center(c){};

  mc_basis_typ(const mc_basis_typ& m, const Point& c)
      : alpha(m.alpha), norm(m.norm), center(c){};
  
  int znum;
  std::vector<double> alpha, norm;
  Point center;
};

class Base_Weight {
 public:
  Base_Weight(const MPI_info&, const Molecule&, const std::string&);

  std::vector<mc_basis_typ> mcBasisList;
  std::vector<double> cum_sum;
  int mc_nprim;

 protected:
  virtual void normalize() = 0;
  void read(const MPI_info&, const Molecule&, const std::string&);
};

class Electron_Pair_Base_Weight : public Base_Weight {
 public:
  Electron_Pair_Base_Weight(const MPI_info&, const Molecule&, const std::string&);
  std::vector<std::array<long int, 4>> cum_sum_index;

  virtual double weight(const Point&, const Point&) const = 0;
  virtual void normalize() = 0;
 protected:
};

class Electron_Pair_GTO_Weight : public Electron_Pair_Base_Weight {
 public:
  Electron_Pair_GTO_Weight(const MPI_info&, const Molecule&, const std::string&);
  double weight(const Point&, const Point&) const override;

 public:
  void normalize() override;
};

class Electron_Base_Weight : public Base_Weight {
 public:
  Electron_Base_Weight(const MPI_info&, const Molecule&, const std::string&);
  std::vector<std::array<long int, 2>> cum_sum_index;

  virtual double weight(const Point&) const = 0;
  virtual void normalize() = 0;
 protected:
};

class Electron_GTO_Weight : public Electron_Base_Weight {
 public:
  Electron_GTO_Weight(const MPI_info&, const Molecule&, const std::string&);
  double weight(const Point&) const override;

 public:
  void normalize() override;
};
#endif

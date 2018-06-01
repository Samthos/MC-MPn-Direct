#pragma once
#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include "qc_geom.h"
#include "qc_mpi.h"

#include <array>
#include <string>
#include <vector>

#ifndef MC_BASIS_T
#define MC_BASIS_T
struct mc_basis_typ {
  int znum;
  std::vector<double> alpha, norm;
  std::array<double, 3> center;
  mc_basis_typ(){};
  mc_basis_typ(const std::vector<double>& a, const std::vector<double>& n,
               const std::array<double, 3>& c)
      : alpha(a), norm(n), center(c){};
  mc_basis_typ(const mc_basis_typ& m, const std::array<double, 3>& c)
      : alpha(m.alpha), norm(m.norm), center(c){};
  mc_basis_typ(const mc_basis_typ& m, const double c[])
      : alpha(m.alpha), norm(m.norm) {
    std::copy(c, c + 3, center.begin());
  };
};

class Base_Weight {
 protected:
  // protected variables

  // protected member functions
  virtual double normalize() = 0;

 public:
  // public variables
  std::vector<mc_basis_typ> mcBasisList;
  std::vector<double> cum_sum;
  std::vector<std::array<long int, 4>> cum_sum_index;

  int mc_nprim;
  void read(const MPI_info&, const Molec&, const std::string&);
};

class GTO_Weight : public Base_Weight {
 private:
  double normalize();

 public:
  double weight(const std::array<double, 3>&,
                const std::array<double, 3>&) const;
  double update();
};
#endif

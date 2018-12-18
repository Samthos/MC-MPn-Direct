// Copyright 2017

#include <cmath>
#include "../qc_geom.h"
#include "../qc_input.h"
#include "../qc_mpi.h"
#include "../el_pair.h"

#ifndef QC_BASIS_H_
#define QC_BASIS_H_
struct BasisMetaData{
  int angular_moment;
  int contraction_begin;
  int contraction_end;
  int ao_begin;
  double pos[3];
};

struct BasisData {
  double *contraction_exp;
  double *contraction_coef;
  double *ao_amplitudes;  // stores AO amplidutes
  double *nw_co;  // obital coefs from nwchem

  double *psi1, *psi2, *occ1, *occ2, *vir1, *vir2;
  double *psiTau1, *psiTau2, *occTau1, *occTau2, *virTau1, *virTau2;

  BasisMetaData *meta_data;
};

class Basis {
 public:
  Basis(IOPs &, MPI_info &, Molec &);
  ~Basis();
  Basis(const Basis&);
  Basis &operator=(const Basis);
  friend void swap(Basis&, Basis&);

  // read write
  void gpu_alloc(int, Molec &);
  void gpu_free();

  // get psi vals
  void host_psi_get(std::vector<el_pair_typ>& el_pair);
  void host_cgs_get(const std::array<double, 3>&, const int);
  void device_psi_get(double *, double *, double *, double *, double *, double *, double *, int);

  // basis set info
  int mc_pair_num;
  int iocc1, iocc2, ivir1, ivir2;
  int qc_nbf;      // number basis functions
  int nShells;      // number of shells???
  int nPrimatives;      // nubmer of primatives
  bool lspherical;  // true if sperical

  BasisData h_basis, d_basis;

  // from nwchem
  int nw_nbf;     // number of basis fcns
  int nw_nmo;  // number of basis fcns in basis set i
  double *nw_en;  // orbital energies from nwchem

 private:
  static constexpr double cf[] = {sqrt(2.5) * 0.5, sqrt(2.5) * 1.5, sqrt(15.0), sqrt(1.5) * 0.5, sqrt(6.0), 1.5, sqrt(15.0) * 0.5};
  static constexpr double cg[] = {2.9580398915498085, 6.2749501990055672, 2.0916500663351894, 1.1180339887498949, 6.7082039324993694,
      2.3717082451262845, 3.1622776601683795, 0.55901699437494745, 3.3541019662496847, 0.73950997288745213, 4.4370598373247132};

  void read(IOPs &, MPI_info &, Molec &);
  void nw_vectors_read(IOPs &, MPI_info &, Molec &);
  void normalize();
};
#endif  // QC_BASIS_H_

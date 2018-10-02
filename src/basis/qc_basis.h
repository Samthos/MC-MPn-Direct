// Copyright 2017

#include "../qc_geom.h"
#include "../qc_input.h"
#include "../qc_mpi.h"

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

  BasisMetaData *meta_data;
};

class Basis {
 private:
  double ang[15], cf[7], cg[11];

 public:
  Basis();
  ~Basis();
  Basis(const Basis &);
  Basis &operator=(const Basis &);

  // read write
  void read(IOPs &, MPI_info &, Molec &);
  void normalize();
  void nw_vectors_read(MPI_info &, Molec &, IOPs &);
  void gpu_alloc(int, Molec &);
  void gpu_free();

  // get psi vals
  void host_psi_get(double *, double *, Molec &);
  void host_cgs_get(double *, Molec &);
  void device_psi_get(double *, double *, double *, double *, double *, double *, double *, int);

  // basis set info
  int iocc1, iocc2, ivir1, ivir2;
  int qc_ncgs;      // number of contracted gausians
  int qc_ngfs;      // number of gasusian functions??
  int qc_nshl;      // number of shells???
  int qc_nprm;      // nubmer of primatives
  bool lspherical;  // true if sperical

  BasisData h_basis, d_basis;

  // from nwchem
  int nw_nbf;     // number of basis fcns
  int nw_nmo;  // number of basis fcns in basis set i
  double *nw_en;  // orbital energies from nwchem
};
#endif  // QC_BASIS_H_

// Copyright 2017

#include "qc_geom.h"
#include "qc_input.h"
#include "qc_mpi.h"

#ifndef QC_BASIS_H_
#define QC_BASIS_H_
struct BasisData {
  int iocc1, iocc2, ivir1, ivir2;
  int qc_ncgs;      // number of contracted gausians
  int qc_ngfs;      // number of gasusian functions??
  int qc_nshl;      // number of shells???
  int qc_nprm;      // nubmer of primatives
  bool lspherical;  // true if sperical

  double *alpha;
  double *norm;
  double *icgs;  // stores AO amplidutes
  int *am;
  int *at;
  int *stop_list;
  int *isgs;

  double *nw_co;  // obital coefs from nwchem

  double *pos;
  double *apos;
};

class Basis {
 private:
  double ang[15], cd[2], cf[7], cg[11];

 public:
  Basis();
  ~Basis();
  Basis(const Basis &);
  Basis &operator=(const Basis &);

  // read write
  void read(IOPs &, MPI_info &, Molec &);
  void new_read2();
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
  int nw_nsets;   // numbe of basis sets
  int nw_nbf;     // number of basis fcns
  int nw_nmo[2];  // number of basis fcns in basis set i
  int nw_iocc;    // number of occpied orbitals
  int nw_icore;   // number of core orbitals
  double *nw_en;  // orbital energies from nwchem
};
#endif  // QC_BASIS_H_

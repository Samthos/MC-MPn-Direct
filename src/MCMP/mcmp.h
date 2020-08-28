#ifndef MCMP_H_
#define MCMP_H_

#include "../qc_monte.h"

#include "mp_functional.h"


template <template <typename, typename> typename Container, template <typename> typename Allocator>
class MCMP : public QC_monte<Container, Allocator> {
 protected:
  typedef Standard_MP_Functional<Container, Allocator> Standard_MP_Functional_Type;
  typedef Direct_MP_Functional<Container, Allocator> Direct_MP_Functional_Type;
  typedef F12_MP_Functional<Container, Allocator> F12_MP_Functional_Type;

 public:
  MCMP(MPI_info p1, IOPs p2, Molecule p3, Basis_Host p4);
  ~MCMP();

  void monte_energy() override;

 protected:
  std::vector<MP_Functional*> energy_functions;
  std::vector<Accumulator*> cv;

  std::vector<double> emp;
  std::vector<std::vector<double>> control;

  void zero_energies();
  virtual void energy();
};
template class MCMP<std::vector, std::allocator>;

#ifdef HAVE_CUDA
template <> void MCMP<thrust::device_vector, thrust::device_allocator>::energy();
template class MCMP<thrust::device_vector, thrust::device_allocator>;

class GPU_MCMP : public MCMP<thrust::device_vector, thrust::device_allocator> {
 public:
  GPU_MCMP(MPI_info p1, IOPs p2, Molecule p3, Basis_Host p4);

 protected:
  void energy() override;
};
#endif

class Dimer : public MCMP<std::vector, std::allocator> {
 public:
  Dimer(MPI_info p1, IOPs p2, Molecule p3, Basis_Host p4);
  ~Dimer();

 protected:
  Tau* monomer_a_tau;
  Tau* monomer_b_tau;
  std::unordered_map<int, Wavefunction_Type> monomer_a_wavefunctions;
  std::unordered_map<int, Wavefunction_Type> monomer_b_wavefunctions;
  void update_wavefunction() override;
  void energy() override;

  template <class Binary_Op> 
  void local_energy(std::unordered_map<int, Wavefunction_Type>& l_wavefunction, Tau* l_tau, Binary_Op);

  std::vector<double> l_emp;
  std::vector<std::vector<double>> l_control;
};

#endif  // MCMP_H_

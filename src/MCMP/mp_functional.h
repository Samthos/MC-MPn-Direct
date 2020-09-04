#ifndef MBPT_H_
#define MBPT_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif
#include <vector>
#include <unordered_map>

#include "../qc_ovps.h"
#include "tau.h"
#include "electron_pair_list.h"
#include "electron_list.h"

namespace MP_FUNCTIONAL_TYPE {
  enum MP_FUNCTIONAL_TYPE {
    STANDARD,
    DIRECT,
    F12,
  };
}

class MP_Functional {
 public:
  MP_Functional(int ncv, int ntc, const std::string& e, MP_FUNCTIONAL_TYPE::MP_FUNCTIONAL_TYPE f) : n_control_variates(ncv), n_tau_coordinates(ntc), extension(e), functional_type(f) {}
  virtual ~MP_Functional() = default;

  int n_control_variates;
  int n_tau_coordinates;
  std::string extension;
  MP_FUNCTIONAL_TYPE::MP_FUNCTIONAL_TYPE functional_type;
};

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
class Standard_MP_Functional : public MP_Functional {
 protected:
  typedef Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
  typedef OVPS<Container, Allocator> OVPS_Type;

 public:
  Standard_MP_Functional(int ncv, int ntc, const std::string& e) : MP_Functional(ncv, ntc, e, MP_FUNCTIONAL_TYPE::STANDARD) {}
  virtual void energy(double& emp, std::vector<double>& control, OVPS_Type&, Electron_Pair_List_Type*, Tau*) = 0;
};

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
class Direct_MP_Functional : public MP_Functional {
 protected:
  typedef Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
  typedef Wavefunction<Container, Allocator> Wavefunction_Type;

 public:
  Direct_MP_Functional(int ncv, int ntc, const std::string& e) : MP_Functional(ncv, ntc, e, MP_FUNCTIONAL_TYPE::DIRECT) {}
  virtual void energy(double& emp, std::vector<double>& control, Wavefunction_Type& psi1, Wavefunction_Type& psi2, Electron_Pair_List_Type*, Tau*) = 0;
};

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
class F12_MP_Functional : public MP_Functional {
 protected:
  typedef Electron_Pair_List_Host Electron_Pair_List_Type;
  typedef Electron_List_Host Electron_List_Type;
  typedef Wavefunction<Container, Allocator> Wavefunction_Type;

 public:
  F12_MP_Functional(int ncv, int ntc, const std::string& e) : MP_Functional(ncv, ntc, e, MP_FUNCTIONAL_TYPE::F12) {}
  virtual void energy(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) = 0;
};

#endif  // MBPT_H_

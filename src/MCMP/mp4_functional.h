#ifndef MP4_Functional_H_
#define MP4_Functional_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#endif
#include "mp_functional.h"

template <int CVMP4, template <typename, typename> typename Container, template <typename> typename Allocator>
class MP4_Functional : public Standard_MP_Functional<Container, Allocator> {
  typedef Container<double, Allocator<double>> vector_double;
  typedef Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
  typedef OVPS<Container, Allocator> OVPS_Type;

 public:
  MP4_Functional(int electron_pairs);
  void energy(double& emp, std::vector<double>& control, OVPS_Type&, Electron_Pair_List_Type*, Tau*) override;

 private:
  void contract(vector_double& result, const vector_double& A, bool A_trans, const vector_double& B, const vector_double& v);
  void mcmp4_ij_helper(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const vector_double& ik, const vector_double& jk,
      const vector_double& il, const vector_double& jl);
  void mcmp4_ij_helper_t1(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const vector_double& ik_1, const vector_double& ik_2, const vector_double& ik_3,
      const vector_double& jk,
      const vector_double& il,
      const vector_double& jl_1, const vector_double& jl_2, const vector_double& jl_3);
  void mcmp4_ij_helper_t2(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const vector_double& ik_1, const vector_double& ik_2,
      const vector_double& jk_1, const vector_double& jk_2,
      const vector_double& il_1, const vector_double& il_2,
      const vector_double& jl_1, const vector_double& jl_2);
  void mcmp4_ij_helper_t3(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const vector_double& ik,
      const vector_double& jk_1, const vector_double& jk_2, const vector_double& jk_3,
      const vector_double& il_1, const vector_double& il_2, const vector_double& il_3,
      const vector_double& jl);
  void mcmp4_energy_ij_fast(double& emp4, std::vector<double>& control, const OVPS_Type& ovps);
  void mcmp4_energy_ik_fast(double& emp4, std::vector<double>& control, const OVPS_Type& ovps);

  void mcmp4_il_helper(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const vector_double& ij, const vector_double& jl,
      const vector_double& ik, const vector_double& kl);
  void mcmp4_il_helper_t1(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const vector_double& ij_1, const vector_double& ij_2, const vector_double& ij_3,
      const vector_double& jl,
      const vector_double& ik,
      const vector_double& kl_1, const vector_double& kl_2, const vector_double& kl_3);
  void mcmp4_il_helper_t2(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const vector_double& ij_1, const vector_double& ij_2,
      const vector_double& jl_1, const vector_double& jl_2,
      const vector_double& ik_1, const vector_double& ik_2,
      const vector_double& kl_1, const vector_double& kl_2);
  void mcmp4_il_helper_t3(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const vector_double& ij,
      const vector_double& jl_1, const vector_double& jl_2, const vector_double& jl_3,
      const vector_double& ik_1, const vector_double& ik_2, const vector_double& ik_3,
      const vector_double& kl);
  void mcmp4_energy_il_fast(double& emp4, std::vector<double>& control, const OVPS_Type& ovps);

  std::array<double, 4> contract_jk(int walker,
      const vector_double& T,
      const vector_double& jk, const vector_double& ik, const vector_double& ij);
  void mcmp4_energy_ijkl_helper(double& emp4, std::vector<double>& control, int offset,
      const std::vector<double>& constants,
      const std::vector<const vector_double*>& ij,
      const std::vector<const vector_double*>& ik,
      const vector_double& il,
      const std::vector<const vector_double*>& jk,
      const vector_double& jl,
      const vector_double& kl);
  void mcmp4_energy_ijkl_t1(double& emp4, std::vector<double>& control,
      const std::vector<double> constants,
      const std::vector<const vector_double*> ij,
      const std::vector<const vector_double*> ik,
      const vector_double& il_1, const vector_double& il_2,
      const std::vector<const vector_double*> jk_1, const std::vector<const vector_double*> jk_2,
      const vector_double& jl,
      const vector_double& kl);
  void mcmp4_energy_ijkl_t2(double& emp4, std::vector<double>& control,
      const std::vector<double> constants,
      const std::vector<const vector_double*> ij,
      const std::vector<const vector_double*> ik_1, const std::vector<const vector_double*> ik_2,
      const vector_double& il,
      const std::vector<const vector_double*> jk,
      const vector_double& jl_1, const vector_double& jl_2,
      const vector_double& kl);
  void mcmp4_energy_ijkl_t3(double& emp4, std::vector<double>& control,
      const std::vector<double> constants,
      const std::vector<const vector_double*> ij_1, const std::vector<const vector_double*> ij_2,
      const std::vector<const vector_double*> ik,
      const vector_double& il,
      const std::vector<const vector_double*> jk,
      const vector_double& jl,
      const vector_double& kl_1, const vector_double& kl_2);
  void mcmp4_energy_ijkl_fast(double& emp4, std::vector<double>& control, const OVPS_Type& ovps);

  int mpn;
  vector_double* rv;
  vector_double* inverse_weight;
  vector_double r_r;
  vector_double r_w;
  vector_double w_w;
  vector_double en_r;
  vector_double en_w;

  vector_double ij_;
  vector_double ik_;
  vector_double il_;
  vector_double jk_;
  vector_double jl_;
  vector_double kl_;

  std::vector<const vector_double*> ext_ptr;
  std::vector<vector_double> ext_data;

  vector_double i_kl;
  vector_double j_kl;
  vector_double ij_rk;
  vector_double ij_wk;
  vector_double ij_rl;
  vector_double ij_wl;
  vector_double T_r;
  vector_double T_w;
  vector_double Av;
};

template class MP4_Functional<0, std::vector, std::allocator>;
template class MP4_Functional<1, std::vector, std::allocator>;
template class MP4_Functional<2, std::vector, std::allocator>;
template class MP4_Functional<3, std::vector, std::allocator>;
template class MP4_Functional<4, std::vector, std::allocator>;
#ifdef HAVE_CUDA
template class MP4_Functional<0, thrust::device_vector, thrust::device_allocator>;
template class MP4_Functional<1, thrust::device_vector, thrust::device_allocator>;
template class MP4_Functional<2, thrust::device_vector, thrust::device_allocator>;
template class MP4_Functional<3, thrust::device_vector, thrust::device_allocator>;
template class MP4_Functional<4, thrust::device_vector, thrust::device_allocator>;
#endif

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
MP_Functional* create_MP4_Functional(int cv_level, int electron_pairs) {
  MP_Functional* mcmp = nullptr;
  if (cv_level == 0) {
    mcmp = new MP4_Functional<0, Container, Allocator>(electron_pairs);
  } else if (cv_level == 1) {
    mcmp = new MP4_Functional<1, Container, Allocator>(electron_pairs);
  } else if (cv_level == 2) {
    mcmp = new MP4_Functional<2, Container, Allocator>(electron_pairs);
  } else if (cv_level == 3) {
    mcmp = new MP4_Functional<3, Container, Allocator>(electron_pairs);
  } else if (cv_level == 4) {
    mcmp = new MP4_Functional<4, Container, Allocator>(electron_pairs);
  }
  
  if (mcmp == nullptr) {
    std::cerr << "MP4_Functional not supported with cv level " << cv_level << "\n";
    exit(0);
  }
  return mcmp;
}
#endif  // MP4_Functional_H_

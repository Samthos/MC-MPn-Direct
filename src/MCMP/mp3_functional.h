#ifndef MP3_Functional_H_
#define MP3_Functional_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#endif
#include "mp_functional.h"

template <int CVMP3, template <typename, typename> typename Container, template <typename> typename Allocator>
class MP3_Functional : public Standard_MP_Functional<Container, Allocator> {
  typedef Container<double, Allocator<double>> vector_double;
  typedef Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
  typedef OVPS<Container, Allocator> OVPS_Type;

 public:
  explicit MP3_Functional(int);
  void energy(double& emp, std::vector<double>& control, OVPS_Type&, Electron_Pair_List_Type*, Tau*) override;

 private:
  void mcmp3_helper(
    const int offset, double constant,
    vector_double& A_ij_1, vector_double& A_ij_2,
    vector_double& A_ik_1, vector_double& A_ik_2,
    vector_double& A_jk_1, vector_double& A_jk_2,
    vector_double& rv, vector_double& wgt);

  void call_helper(OVPS_Type& ovps, vector_double& j_op, vector_double& k_op, int offset);

  int electron_pairs;
  vector_double A_ij;
  vector_double A_ik;
  vector_double A_jk;
  vector_double A_i;
  vector_double A;
  vector_double ctrl;
};

template class MP3_Functional<0, std::vector, std::allocator>;
template class MP3_Functional<1, std::vector, std::allocator>;
template class MP3_Functional<2, std::vector, std::allocator>;
template class MP3_Functional<3, std::vector, std::allocator>;
#ifdef HAVE_CUDA
template class MP3_Functional<0, thrust::device_vector, thrust::device_allocator>;
template class MP3_Functional<1, thrust::device_vector, thrust::device_allocator>;
template class MP3_Functional<2, thrust::device_vector, thrust::device_allocator>;
template class MP3_Functional<3, thrust::device_vector, thrust::device_allocator>;
#endif

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
MP_Functional* create_MP3_Functional(int cv_level, int electron_pairs) { 
  MP_Functional* mcmp = nullptr;
  if (cv_level == 0) {
    mcmp = new MP3_Functional<0, Container, Allocator>(electron_pairs);
  } else if (cv_level == 1) {
    mcmp = new MP3_Functional<1, Container, Allocator>(electron_pairs);
  } else if (cv_level == 2) {
    mcmp = new MP3_Functional<2, Container, Allocator>(electron_pairs);
  } else if (cv_level == 3) {
    mcmp = new MP3_Functional<3, Container, Allocator>(electron_pairs);
  }
  
  if (mcmp == nullptr) {
    std::cerr << "MP3_Functional not supported with cv level " << cv_level << "\n";
    exit(0);
  }
  return mcmp;
}
#endif  // MP3_Functional_H_

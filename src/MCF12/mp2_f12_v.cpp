#include "mp2_f12_v.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
MP2_F12_V<Container, Allocator>::MP2_F12_V(const IOPs& iops, std::string extension) :
    F12_MP_Functional<Container, Allocator>(0, 0, extension),
    traces(iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRONS]),
    T_ip_io(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    T_ip_jo(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    T_io_jo(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    T_ip(iops.iopns[KEYS::ELECTRON_PAIRS], 0.0),
    T_io(iops.iopns[KEYS::ELECTRONS], 0.0),
    correlation_factor(create_Correlation_Factor_Data<Container, Allocator>(
          iops.iopns[KEYS::ELECTRONS],
          iops.iopns[KEYS::ELECTRON_PAIRS],
          static_cast<CORRELATION_FACTOR::Type>(iops.iopns[KEYS::F12_CORRELATION_FACTOR]),
          iops.dopns[KEYS::F12_GAMMA],
          iops.dopns[KEYS::F12_BETA]))
{
  nsamp_pair = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp_one_1 = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]);
  nsamp_one_2 = nsamp_one_1 / static_cast<double>(iops.iopns[KEYS::ELECTRONS] - 1.0);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
MP2_F12_V<Container, Allocator>::~MP2_F12_V() {
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void MP2_F12_V<Container, Allocator>::energy(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  calculate_v(emp, control, wavefunctions, electron_pair_list, electron_list);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void MP2_F12_V<Container, Allocator>::calculate_v_4e_help(
    vector_double& R_ip_io, vector_double& R_ip_jo, 
    const vector_double& S_io_jo,
    const vector_double& S_ip_io_1, const vector_double& S_ip_io_2,
    const vector_double& S_ip_jo_1, const vector_double& S_ip_jo_2,
    const vector_double& S_ip,
    double alpha, double beta,
    size_t size, size_t size_ep) {
  blas_wrapper.multiplies(S_ip_jo_1.begin(), S_ip_jo_1.end(), S_ip_jo_2.begin(), R_ip_jo.begin());
  blas_wrapper.dgemm(true, false,
      size, size_ep, size,
      1.0,
      S_io_jo, size,
      R_ip_jo, size,
      0.0,
      R_ip_io, size);
  blas_wrapper.multiplies(S_ip_io_1.begin(), S_ip_io_1.end(), S_ip_io_2.begin(), R_ip_jo.begin());
  blas_wrapper.multiplies(R_ip_jo.begin(), R_ip_jo.end(), R_ip_io.begin(), R_ip_jo.begin());
  blas_wrapper.dgemv(
      false,
      size, size_ep,
      alpha,
      R_ip_jo, size,
      S_ip, 1,
      beta,
      T_io, 1);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double MP2_F12_V<Container, Allocator>::calculate_v_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  // f12 / (r12 w12)
  blas_wrapper.dgekv(electron_pair_list->size(), 
      1.0, 
      correlation_factor->f12p, 1, 
      electron_pair_list->rv,  1,
      0.0, 
      T_ip_io, 1);
  // c1 p11 p22
  blas_wrapper.dgekv(electron_pair_list->size(), 
      c1, 
      traces.p11, 0, 1, 
      traces.p22, 0, 1,
      0.0, 
      T_ip_io, electron_pair_list->size(), 1);
  // c1 p11 p22 + c2 p12 p21
  blas_wrapper.dgekv(electron_pair_list->size(),
      c2, 
      traces.p12, 0, 1,
      traces.p12, 0, 1,
      1.0, 
      T_ip_io, electron_pair_list->size(), 1);
  // c1 f12 p11 p22 / (r12 w12) + c2 f12 p12 p21 / (r12 w12)
  double result = blas_wrapper.ddot(electron_pair_list->size(), 
      T_ip_io, 0, 1, 
      T_ip_io, electron_pair_list->size(), 1);
  result *= nsamp_pair;
  return result;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double MP2_F12_V<Container, Allocator>::calculate_v_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  // f23 k31
  blas_wrapper.dgekm(false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      correlation_factor->f23, electron_list->size(),
      traces.k13, electron_list->size(),
      0.0, 
      T_ip_io, electron_list->size());
  // f23 o13 k31
  blas_wrapper.dgekm(false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      T_ip_io, electron_list->size(),
      traces.p13, electron_list->size(),
      0.0, 
      T_ip_jo, electron_list->size());
  // o22 / (r12 w12)
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      electron_pair_list->rv, 1,
      traces.p22, 1,
      0.0, 
      T_ip, 1);
  // c1 f23 o13 o22 k31 / (r12 w12)
  blas_wrapper.dgemv(
      false,
      electron_list->size(), electron_pair_list->size(), 
      c1,
      T_ip_jo, electron_list->size(),
      T_ip, 1,
      0.0,
      T_io, 1);

  // f23 o23 k31
  blas_wrapper.dgekm(false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      T_ip_io, electron_list->size(),
      traces.p23, electron_list->size(),
      0.0, 
      T_ip_jo, electron_list->size());
  // o12 / (r12 w12)
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      electron_pair_list->rv, 1,
      traces.p12, 1,
      0.0, 
      T_ip, 1);
  // f23 (o13 o22 k31 + c2 o12 o23 k31) / (r12 w12)
  blas_wrapper.dgemv(
      false,
      electron_list->size(), electron_pair_list->size(),
      c2,
      T_ip_jo, electron_list->size(),
      T_ip, 1,
      1.0,
      T_io, 1);
  // f23 (o13 o22 k31 + c2 o12 o23 k31) / (r12 w12 w3)
  double result = blas_wrapper.ddot(electron_list->size(),
      T_io, 1,
      electron_list->inverse_weight, 1);
  result = -2.0 * result * nsamp_pair * nsamp_one_1;
  return result;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double MP2_F12_V<Container, Allocator>::calculate_v_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  // f34 / w3 <- might be w4?
  blas_wrapper.ddgmm(BLAS_WRAPPER::RIGHT_SIDE,
      electron_list->size(), electron_list->size(),
      correlation_factor->f12o, electron_list->size(),
      electron_list->inverse_weight, 1,
      T_io_jo, electron_list->size());
  // f34 / (w3 w4)
  blas_wrapper.ddgmm(BLAS_WRAPPER::LEFT_SIDE,
      electron_list->size(), electron_list->size(),
      T_io_jo, electron_list->size(),
      electron_list->inverse_weight, 1,
      T_io_jo, electron_list->size());

  // c1 f34 o13 o24 o31 o42 / (r12 w12 w3 w4)
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.p13, traces.k13,
      traces.p23, traces.k23,
      electron_pair_list->rv,
      c1, 0.0,
      electron_list->size(), electron_pair_list->size());

  // -c1 f34 o13 o24 v31 v42 / (r12 w12 w3 w4)
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.p13, traces.v13,
      traces.p23, traces.v23,
      electron_pair_list->rv,
      -c1, 1.0,
      electron_list->size(), electron_pair_list->size());

  // c2 f34 o14 o23 o31 o42 / (r12 w12 w3 w4)
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo,
      traces.p23, traces.k13,
      traces.p13, traces.k23,
      electron_pair_list->rv,
      c2, 1.0,
      electron_list->size(), electron_pair_list->size());

  // -c2 f34 o14 o23 v31 v42 / (r12 w12 w3 w4)
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.p23, traces.v13,
      traces.p13, traces.v23,
      electron_pair_list->rv,
      -c2, 1.0,
      electron_list->size(), electron_pair_list->size());
  return blas_wrapper.accumulate(T_io.size(), T_io, 1) * nsamp_pair * nsamp_one_2;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void MP2_F12_V<Container, Allocator>::calculate_v(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  traces.update_v(wavefunctions);
  correlation_factor->update(electron_pair_list, electron_list);
  emp += calculate_v_2e(electron_pair_list, electron_list);
  emp += calculate_v_3e(electron_pair_list, electron_list);
  emp += calculate_v_4e(electron_pair_list, electron_list);
}

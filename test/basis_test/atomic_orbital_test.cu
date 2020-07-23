#include <cmath>
#include <thrust/device_vector.h>
#include "gtest/gtest.h"

#include "../../src/basis/atomic_orbital.h"
#include "../../src/basis/shell.h"

  std::ostream& operator << (std::ostream& os, const std::vector<double>& v) {
    for (auto &it : v) {
      os << std::setprecision(16) << it << "\n";
    }
    return os;
  }

namespace HostTests {
  class HostCartesianAtomicOrbitalTest : public testing::TestWithParam<SHELL::Shell_Type> {
    public:
    HostCartesianAtomicOrbitalTest() : shell_type(GetParam()), 
      contraction_amplitudes(1, 1.0),
      contraction_amplitude_derivatives(1, 1.0),
      ao_amplitudes(number_of_cartesian_polynomials(shell_type), 0.0),
      r({0.0, 0.0, 0.0}),
      atomic_orbital(0, 0, 0, 0, shell_type, false, r.data())
    { }

    void derivative_test(int ldx) {
      int idx = 0;
      std::array<int, 3> l{0, 0, 0};
      for (l[0] = shell_type; l[0] >= 0; l[0]--) {
        for (l[1] = shell_type - l[0]; l[1] >= 0; l[1]--, idx++) {
          l[2] = shell_type - l[0] - l[1];
          int n = l[ldx];
          double value = pow(2, n+1) + n * pow(2, n-1);
          ASSERT_FLOAT_EQ(value, ao_amplitudes[idx]);
        }
      }
    }

    SHELL::Shell_Type shell_type;
    std::vector<double> contraction_amplitudes;
    std::vector<double> contraction_amplitude_derivatives;
    std::vector<double> ao_amplitudes;
    std::array<double, 3> r;
    Atomic_Orbital atomic_orbital;
  };

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAO) {
    std::vector<double> data = {
      1, 2, 3, 4, 6, 9, 8, 12, 18, 27, 16, 24, 36, 54, 81};

    double s[] = {1.0, 2.0, 3.0};
    atomic_orbital.evaluate_ao(ao_amplitudes.data(), contraction_amplitudes.data(), s);

    for (int i = 0; i < ao_amplitudes.size(); i++) {
      ASSERT_FLOAT_EQ(data[i], ao_amplitudes[i]);
    }
  }

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAODX) {
    double s[] = {2.0, 1.0, 1.0};
    atomic_orbital.evaluate_ao_dx(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    derivative_test(0);
  }

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAODY) {
    double s[] = {1.0, 2.0, 1.0};
    atomic_orbital.evaluate_ao_dy(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    derivative_test(1);
  }

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAODZ) {
    double s[] = {1.0, 1.0, 2.0};
    atomic_orbital.evaluate_ao_dz(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    derivative_test(2);
  }

  INSTANTIATE_TEST_SUITE_P(InstantiationName,
                         HostCartesianAtomicOrbitalTest,
                         testing::Values(SHELL::S, SHELL::P, SHELL::D, SHELL::F, SHELL::G));
}

namespace DeviceTests {
  class DeviceCartesianAtomicOrbitalTest : public testing::TestWithParam<SHELL::Shell_Type> {
    public:
    DeviceCartesianAtomicOrbitalTest() : shell_type(GetParam()), 
      contraction_amplitudes(1, 1.0),
      contraction_amplitude_derivatives(1, 1.0),
      ao_amplitudes(number_of_cartesian_polynomials(shell_type)),
      h_ao_amplitudes(number_of_cartesian_polynomials(shell_type))
    { 
      std::array<double,3 > r{0.0, 0.0, 0.0};
      std::vector<Atomic_Orbital> h_atomic_orbital;
      h_atomic_orbital.emplace_back(0, 0, 0, 0, shell_type, false, r.data());

      atomic_orbital.resize(h_atomic_orbital.size());
      thrust::copy(h_atomic_orbital.begin(), h_atomic_orbital.end(), atomic_orbital.begin());
    }

    void derivative_test(int ldx) {
      thrust::copy(ao_amplitudes.begin(), ao_amplitudes.end(), h_ao_amplitudes.begin());

      int idx = 0;
      std::array<int, 3> l{0, 0, 0};
      for (l[0] = shell_type; l[0] >= 0; l[0]--) {
        for (l[1] = shell_type - l[0]; l[1] >= 0; l[1]--, idx++) {
          l[2] = shell_type - l[0] - l[1];
          int n = l[ldx];
          double value = pow(2, n+1) + n * pow(2, n-1);
          ASSERT_FLOAT_EQ(value, ao_amplitudes[idx]);
        }
      }
    }

    SHELL::Shell_Type shell_type;
    thrust::device_vector<double> contraction_amplitudes;
    thrust::device_vector<double> contraction_amplitude_derivatives;
    thrust::device_vector<double> ao_amplitudes;
    std::vector<double> h_ao_amplitudes;
    std::array<double, 3> r;
    thrust::device_vector<Atomic_Orbital> atomic_orbital;
  };

  __global__ 
  void call_evaluate_ao(Atomic_Orbital* atomic_orbitals, double* ao_amplitudes, double* contraction_amplitudes, double x[3]) {
    atomic_orbitals->evaluate_ao(ao_amplitudes, contraction_amplitudes, x);
  }

  TEST_P(DeviceCartesianAtomicOrbitalTest, evaluateCartesianAO) {
    std::vector<double> data = {
      1, 2, 3, 4, 6, 9, 8, 12, 18, 27, 16, 24, 36, 54, 81};

    std::vector<double> h_s({1.0, 2.0, 3.0});
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao<<<1,1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), d_s.data().get());

    thrust::copy(ao_amplitudes.begin(), ao_amplitudes.end(), h_ao_amplitudes.begin());
    for (int i = 0; i < ao_amplitudes.size(); i++) {
      ASSERT_FLOAT_EQ(data[i], h_ao_amplitudes[i]);
    }
  }

  __global__ 
  void call_evaluate_ao_dx(Atomic_Orbital* atomic_orbitals, double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, double x[3]) {
    atomic_orbitals->evaluate_ao_dx(ao_amplitudes, contraction_amplitudes, contraction_amplitude_derivatives, x);
  }

  TEST_P(DeviceCartesianAtomicOrbitalTest, evaluateCartesianAODX) {
    std::vector<double> h_s({2.0, 1.0, 1.0});
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao_dx<<<1, 1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), contraction_amplitude_derivatives.data().get(), d_s.data().get());
    derivative_test(0);
  }

  __global__ 
  void call_evaluate_ao_dy(Atomic_Orbital* atomic_orbitals, double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, double x[3]) {
    atomic_orbitals->evaluate_ao_dy(ao_amplitudes, contraction_amplitudes, contraction_amplitude_derivatives, x);
  }

  TEST_P(DeviceCartesianAtomicOrbitalTest, evaluateCartesianAODY) {
    std::vector<double> h_s({1.0, 2.0, 1.0});
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao_dy<<<1, 1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), contraction_amplitude_derivatives.data().get(), d_s.data().get());
    derivative_test(1);
  }

  __global__ 
  void call_evaluate_ao_dz(Atomic_Orbital* atomic_orbitals, double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, double x[3]) {
    atomic_orbitals->evaluate_ao_dz(ao_amplitudes, contraction_amplitudes, contraction_amplitude_derivatives, x);
  }

  TEST_P(DeviceCartesianAtomicOrbitalTest, evaluateCartesianAODZ) {
    std::vector<double> h_s({1.0, 1.0, 2.0});
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao_dz<<<1, 1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), contraction_amplitude_derivatives.data().get(), d_s.data().get());
    derivative_test(2);
  }

  INSTANTIATE_TEST_SUITE_P(InstantiationName,
                         DeviceCartesianAtomicOrbitalTest,
                         testing::Values(SHELL::S, SHELL::P, SHELL::D, SHELL::F, SHELL::G));
}


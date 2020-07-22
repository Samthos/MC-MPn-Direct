#include <cmath>
#include "gtest/gtest.h"

#include "../../src/basis/atomic_orbital.h"
#include "../../src/basis/shell.h"

namespace {
  class HostCartesianAtomicOrbitalTest : public testing::TestWithParam<SHELL::Shell_Type> {
    public:
    HostCartesianAtomicOrbitalTest() : shell_type(GetParam()), 
      contraction_amplitudes(1, 1.0),
      contraction_amplitude_derivatives(1, 1.0),
      ao_amplitudes(number_of_cartesian_polynomials(shell_type), 0.0),
      r({0.0, 0.0, 0.0}),
      s({1.0, 2.0, 3.0}),
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
    std::array<double, 3> s;
    Atomic_Orbital atomic_orbital;
  };

  std::ostream& operator << (std::ostream& os, const std::vector<double>& v) {
    for (auto &it : v) {
      os << std::setprecision(16) << it << "\n";
    }
    return os;
  }

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAO) {
    std::vector<double> data = {
      1, 2, 3, 4, 6, 9, 8, 12, 18, 27, 16, 24, 36, 54, 81};

    std::array<double, 3> s{1.0, 2.0, 3.0};
    atomic_orbital.evaluate_ao(ao_amplitudes.data(), contraction_amplitudes.data(), s);

    for (int i = 0; i < ao_amplitudes.size(); i++) {
      ASSERT_FLOAT_EQ(data[i], ao_amplitudes[i]);
    }
  }

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAODX) {
    std::array<double, 3> s{2.0, 1.0, 1.0};
    atomic_orbital.evaluate_ao_dx(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    derivative_test(0);
  }

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAODY) {
    std::array<double, 3> s{1.0, 2.0, 1.0};
    atomic_orbital.evaluate_ao_dy(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    derivative_test(1);
  }

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAODZ) {
    std::array<double, 3> s{1.0, 1.0, 2.0};
    atomic_orbital.evaluate_ao_dz(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    derivative_test(2);
  }

  INSTANTIATE_TEST_SUITE_P(InstantiationName,
                         HostCartesianAtomicOrbitalTest,
                         testing::Values(SHELL::S, SHELL::P, SHELL::D, SHELL::F, SHELL::G));
}


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

namespace ContractionHostTests {
  std::vector<double> make_contraction_coeficients(int n) {
    std::vector<double> v(n);
    for (int i = 0; i < v.size(); i++) {
      v[i] = i + 1;
    }
    return v;
  }

  std::vector<double> make_contraction_exponents(int n) {
    std::vector<double> v(n);
    for (int i = 0; i < v.size(); i++) {
      v[i] = log(i + 1) / 3.0;
    }
    return v;
  }

  std::vector<Atomic_Orbital> make_atomic_orbital() {
    std::vector<Atomic_Orbital> atomic_orbital;
    std::array<double, 3> r{0.0, 0.0, 0.0};
    atomic_orbital.emplace_back(0, 10, 0, 0, SHELL::S, false, r.data());
    return atomic_orbital;
  }

  class HostContractionAtomicOrbitalTest : public testing::Test {
    public:
    HostContractionAtomicOrbitalTest() :
      contraction_coeficients(make_contraction_coeficients(10)),
      contraction_exponents(make_contraction_exponents(10)),
      contraction_amplitudes(1),
      contraction_amplitude_derivatives(1),
      atomic_orbital(make_atomic_orbital())
    {
      contraction_result = contraction_coeficients.size();
      contraction_derivative_result = 0;
      for (auto &it : contraction_exponents) {
        contraction_derivative_result += it;
      }
      contraction_derivative_result *= -2.0;
    }

    double contraction_result;
    double contraction_derivative_result;

    std::vector<double> contraction_coeficients;
    std::vector<double> contraction_exponents;
    std::vector<double> contraction_amplitudes;
    std::vector<double> contraction_amplitude_derivatives;
    std::vector<Atomic_Orbital> atomic_orbital;
  };

  TEST_F(HostContractionAtomicOrbitalTest, evaluateContractions) {
    double s[] = {1.0, 1.0, 1.0};
    atomic_orbital[0].evaluate_contraction(contraction_amplitudes.data(), contraction_exponents.data(), contraction_coeficients.data(), s);
    ASSERT_FLOAT_EQ(contraction_result, contraction_amplitudes[0]); 
  }

  TEST_F(HostContractionAtomicOrbitalTest, evaluateContractionsWithDerivatives) {
    double s[] = {1.0, 1.0, 1.0};
    atomic_orbital[0].evaluate_contraction_with_derivative(contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), contraction_exponents.data(), contraction_coeficients.data(), s);
    ASSERT_FLOAT_EQ(contraction_result, contraction_amplitudes[0]); 
    ASSERT_FLOAT_EQ(contraction_derivative_result, contraction_amplitude_derivatives[0]);
  }

  class DeviceContractionAtomicOrbitalTest : public testing::Test {
    public:
    DeviceContractionAtomicOrbitalTest() : 
      h_contraction_amplitudes(1),
      h_contraction_amplitude_derivatives(1),
      contraction_coeficients(make_contraction_coeficients(10)),
      contraction_exponents(make_contraction_exponents(10)),
      contraction_amplitudes(1),
      contraction_amplitude_derivatives(1),
      atomic_orbital(make_atomic_orbital())
    {
      contraction_result = contraction_coeficients.size();
      contraction_derivative_result = 0;
      auto h_contraction_exponents = make_contraction_exponents(10);
      for (auto &it : h_contraction_exponents) {
        contraction_derivative_result += it;
      }
      contraction_derivative_result *= -2.0;
    }

    double contraction_result;
    double contraction_derivative_result;

    std::vector<double> h_contraction_amplitudes;
    std::vector<double> h_contraction_amplitude_derivatives;
    
    thrust::device_vector<double> contraction_coeficients;
    thrust::device_vector<double> contraction_exponents;
    thrust::device_vector<double> contraction_amplitudes;
    thrust::device_vector<double> contraction_amplitude_derivatives;
    thrust::device_vector<Atomic_Orbital> atomic_orbital;
  };

  __global__
  void call_evaluate_contraction(Atomic_Orbital* atomic_orbital, double* contraction_amplitudes, double* contraction_exponents, double* contraction_coeficients, double* s) {
    atomic_orbital->evaluate_contraction(contraction_amplitudes, contraction_exponents, contraction_coeficients, s);
  }

  TEST_F(DeviceContractionAtomicOrbitalTest, evaluateContractions) {
    std::vector<double> h_s = {1.0, 1.0, 1.0};
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_contraction<<<1, 1>>>(atomic_orbital.data().get(), contraction_amplitudes.data().get(), contraction_exponents.data().get(), contraction_coeficients.data().get(), d_s.data().get());

    thrust::copy(contraction_amplitudes.begin(), contraction_amplitudes.end(), h_contraction_amplitudes.begin());
    ASSERT_FLOAT_EQ(10, h_contraction_amplitudes[0]); 
  }

  __global__
  void call_evaluate_contraction_with_derivative(Atomic_Orbital* atomic_orbital, double* contraction_amplitudes, double* contraction_amplitude_derivatives, double* contraction_exponents, double* contraction_coeficients, double* s) {
    atomic_orbital->evaluate_contraction_with_derivative(contraction_amplitudes, contraction_amplitude_derivatives, contraction_exponents, contraction_coeficients, s);
  }

  TEST_F(DeviceContractionAtomicOrbitalTest, evaluateContractionsWithDerivatives) {
    std::vector<double> h_s = {1.0, 1.0, 1.0};
    thrust::device_vector<double> d_s(h_s);

    call_evaluate_contraction_with_derivative<<<1, 1>>>(atomic_orbital.data().get(), contraction_amplitudes.data().get(), contraction_amplitude_derivatives.data().get(), contraction_exponents.data().get(), contraction_coeficients.data().get(), d_s.data().get());

    thrust::copy(contraction_amplitudes.begin(), contraction_amplitudes.end(), h_contraction_amplitudes.begin());
    ASSERT_FLOAT_EQ(10, h_contraction_amplitudes[0]); 

    thrust::copy(contraction_amplitude_derivatives.begin(), contraction_amplitude_derivatives.end(), h_contraction_amplitude_derivatives.begin());
    ASSERT_FLOAT_EQ(contraction_derivative_result, h_contraction_amplitude_derivatives[0]);
  }
}

namespace CartesianHostTests {
  class HostCartesianAtomicOrbitalTest : public testing::TestWithParam<SHELL::Shell_Type> {
    public:
    HostCartesianAtomicOrbitalTest() : shell_type(GetParam()), 
      contraction_amplitudes(1, 1.0),
      contraction_amplitude_derivatives(1, 1.0),
      ao_amplitudes(number_of_cartesian_polynomials(shell_type), 0.0)
    {
      std::array<double, 3> r{0.0, 0.0, 0.0};
      atomic_orbital.emplace_back(0, 0, 0, 0, shell_type, false, r.data());
    }

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
    std::vector<Atomic_Orbital> atomic_orbital;
  };

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAO) {
    std::vector<double> data = {
      1, 2, 3, 4, 6, 9, 8, 12, 18, 27, 16, 24, 36, 54, 81};

    double s[] = {1.0, 2.0, 3.0};
    atomic_orbital[0].evaluate_ao(ao_amplitudes.data(), contraction_amplitudes.data(), s);

    for (int i = 0; i < ao_amplitudes.size(); i++) {
      ASSERT_FLOAT_EQ(data[i], ao_amplitudes[i]);
    }
  }

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAODX) {
    double s[] = {2.0, 1.0, 1.0};
    atomic_orbital[0].evaluate_ao_dx(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    derivative_test(0);
  }

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAODY) {
    double s[] = {1.0, 2.0, 1.0};
    atomic_orbital[0].evaluate_ao_dy(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    derivative_test(1);
  }

  TEST_P(HostCartesianAtomicOrbitalTest, evaluateCartesianAODZ) {
    double s[] = {1.0, 1.0, 2.0};
    atomic_orbital[0].evaluate_ao_dz(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    derivative_test(2);
  }

  INSTANTIATE_TEST_SUITE_P(InstantiationName,
      HostCartesianAtomicOrbitalTest,
      testing::Values(SHELL::S, SHELL::P, SHELL::D, SHELL::F, SHELL::G));
}

namespace CartesianDeviceTests {
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

    std::vector<double> h_s = {1.0, 2.0, 3.0};
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
    std::vector<double> h_s = {2.0, 1.0, 1.0};
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao_dx<<<1, 1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), contraction_amplitude_derivatives.data().get(), d_s.data().get());
    derivative_test(0);
  }

  __global__ 
  void call_evaluate_ao_dy(Atomic_Orbital* atomic_orbitals, double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, double x[3]) {
    atomic_orbitals->evaluate_ao_dy(ao_amplitudes, contraction_amplitudes, contraction_amplitude_derivatives, x);
  }

  TEST_P(DeviceCartesianAtomicOrbitalTest, evaluateCartesianAODY) {
    std::vector<double> h_s = {1.0, 2.0, 1.0};
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao_dy<<<1, 1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), contraction_amplitude_derivatives.data().get(), d_s.data().get());
    derivative_test(1);
  }

  __global__ 
  void call_evaluate_ao_dz(Atomic_Orbital* atomic_orbitals, double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, double x[3]) {
    atomic_orbitals->evaluate_ao_dz(ao_amplitudes, contraction_amplitudes, contraction_amplitude_derivatives, x);
  }

  TEST_P(DeviceCartesianAtomicOrbitalTest, evaluateCartesianAODZ) {
    std::vector<double> h_s = {1.0, 1.0, 2.0};
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao_dz<<<1, 1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), contraction_amplitude_derivatives.data().get(), d_s.data().get());
    derivative_test(2);
  }

  INSTANTIATE_TEST_SUITE_P(InstantiationName,
      DeviceCartesianAtomicOrbitalTest,
      testing::Values(SHELL::S, SHELL::P, SHELL::D, SHELL::F, SHELL::G));
}

namespace SphericalHostTests {
  class HostSphericalAtomicOrbitalTest : public testing::TestWithParam<SHELL::Shell_Type> {
    public:
    HostSphericalAtomicOrbitalTest() : shell_type(GetParam()), 
      contraction_amplitudes(1, 1.0),
      contraction_amplitude_derivatives(1, 1.0),
      ao_amplitudes(number_of_spherical_polynomials(shell_type), 0.0)
    {
      std::array<double, 3> r{0.0, 0.0, 0.0};
      atomic_orbital.emplace_back(0, 0, 0, 0, shell_type, true, r.data());
    }

    void run_asserts(const std::vector<double>& data) {
      for (auto i = 0; i < ao_amplitudes.size(); i++) {
        ASSERT_FLOAT_EQ(data[i], ao_amplitudes[i]) << shell_type << i;
      }
    }

    SHELL::Shell_Type shell_type;
    std::vector<double> contraction_amplitudes;
    std::vector<double> contraction_amplitude_derivatives;
    std::vector<double> ao_amplitudes;
    std::vector<Atomic_Orbital> atomic_orbital;
  };

  TEST_P(HostSphericalAtomicOrbitalTest, evaluateSphericalAO) {
    std::vector<std::vector<double>> data = {
      {1}, 
      {1, 2, 3},
      {3.464101615137754, 10.39230484541326, 6.5, -5.196152422706631, -2.59807621135329},
      {-1.58113883008419, 23.2379000772445, 37.96709101313926, 4.5, -18.98354550656963, -17.42842505793338, 8.696263565463044},
      {-17.74823934929885, -12.54990039801114, 109.5673308974897, 99.61174629530396, -44.625, -49.80587314765198, -82.17549817311728, 69.02445218906124, -5.176569810212166}};

    double s[] = {1.0, 2.0, 3.0};
    atomic_orbital[0].evaluate_ao(ao_amplitudes.data(), contraction_amplitudes.data(), s);
    run_asserts(data[shell_type]);
  }

  TEST_P(HostSphericalAtomicOrbitalTest, evaluateSphericalAODX) {
    std::vector<std::vector<double>> data = {
      {2},
      {5, 2, 2},
      {8.660254037844386, 3.464101615137754, -5, -8.660254037844386, 8.6602540378443},
      {26.87936011143123, 19.36491673103708, -3.674234614174766, -19, 7.960841664045326, 19.36491673103708, -10.27740239554723},
      {68.03491750564559, 71.11610225539643, -3.354101966249679, -26.87936011143123, -6.25, 62.45498378832549, -1.118033988749893, -27.19145086235747, -4.437059837324718}
    };

    double s[] = {2.0, 1.0, 1.0};
    atomic_orbital[0].evaluate_ao_dx(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    run_asserts(data[shell_type]);
  }

  TEST_P(HostSphericalAtomicOrbitalTest, evaluateSphericalAODY) {
    std::vector<std::vector<double>> data = {
      {2},
      {2, 5, 2},
      {8.660254037844386, 8.660254037844386, -5, -3.464101615137754, -8.6602540378443},
      {-10.27740239554723, 19.36491673103708, -7.960841664045326, -19, 3.674234614174766, -19.36491673103708, 26.87936011143123},
      {-68.03491750564559, -27.19145086235747, -3.354101966249679, -62.45498378832549, -6.25, 26.87936011143122, 1.118033988749893, 71.11610225539643, -4.437059837324718}
    };

    double s[] = {1.0, 2.0, 1.0};
    atomic_orbital[0].evaluate_ao_dy(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    run_asserts(data[shell_type]);
  }

  TEST_P(HostSphericalAtomicOrbitalTest, evaluateSphericalAODZ) {
    std::vector<std::vector<double>> data = {
      {2},
      {2, 2, 5},
      {3.464101615137754, 8.660254037844386, 10, -8.660254037844386, 0},
      {3.162277660168379, 19.36491673103708, 26.94438717061496, 13, -26.94438717061496, 0, 3.162277660168379},
      {0, 20.91650066335189, 76.02631123499285, 64.82669203345178, -5, -64.82669203345179, 0, 20.91650066335189, -5.916079783099618}
    };

    double s[] = {1.0, 1.0, 2.0};
    atomic_orbital[0].evaluate_ao_dz(ao_amplitudes.data(), contraction_amplitudes.data(), contraction_amplitude_derivatives.data(), s);
    run_asserts(data[shell_type]);
  }

  INSTANTIATE_TEST_SUITE_P(InstantiationName,
      HostSphericalAtomicOrbitalTest,
      testing::Values(SHELL::S, SHELL::P, SHELL::D, SHELL::F, SHELL::G));
}

namespace SphericalDeviceTests {
  class DeviceSphericalAtomicOrbitalTest : public testing::TestWithParam<SHELL::Shell_Type> {
    public:
    DeviceSphericalAtomicOrbitalTest() : shell_type(GetParam()), 
      contraction_amplitudes(1, 1.0),
      contraction_amplitude_derivatives(1, 1.0),
      ao_amplitudes(number_of_spherical_polynomials(shell_type)),
      h_ao_amplitudes(number_of_spherical_polynomials(shell_type))
    { 
      std::array<double,3 > r{0.0, 0.0, 0.0};
      std::vector<Atomic_Orbital> h_atomic_orbital;
      h_atomic_orbital.emplace_back(0, 0, 0, 0, shell_type, true, r.data());

      atomic_orbital.resize(h_atomic_orbital.size());
      thrust::copy(h_atomic_orbital.begin(), h_atomic_orbital.end(), atomic_orbital.begin());
    }

    void run_asserts(const std::vector<double>& data) {
      thrust::copy(ao_amplitudes.begin(), ao_amplitudes.end(), h_ao_amplitudes.begin());
      for (auto i = 0; i < ao_amplitudes.size(); i++) {
        ASSERT_FLOAT_EQ(data[i], h_ao_amplitudes[i]) << shell_type << i;
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

  TEST_P(DeviceSphericalAtomicOrbitalTest, evaluateSphericalAO) {
    std::vector<std::vector<double>> data = {
      {1}, 
      {1, 2, 3},
      {3.464101615137754, 10.39230484541326, 6.5, -5.196152422706631, -2.59807621135329},
      {-1.58113883008419, 23.2379000772445, 37.96709101313926, 4.5, -18.98354550656963, -17.42842505793338, 8.696263565463044},
      {-17.74823934929885, -12.54990039801114, 109.5673308974897, 99.61174629530396, -44.625, -49.80587314765198, -82.17549817311728, 69.02445218906124, -5.176569810212166}};

    std::vector<double> h_s = {1.0, 2.0, 3.0};
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao<<<1,1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), d_s.data().get());
    run_asserts(data[shell_type]);
  }

  __global__ 
  void call_evaluate_ao_dx(Atomic_Orbital* atomic_orbitals, double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, double x[3]) {
    atomic_orbitals->evaluate_ao_dx(ao_amplitudes, contraction_amplitudes, contraction_amplitude_derivatives, x);
  }

  TEST_P(DeviceSphericalAtomicOrbitalTest, evaluateSphericalAODX) {
    std::vector<std::vector<double>> data = {
      {2},
      {5, 2, 2},
      {8.660254037844386, 3.464101615137754, -5, -8.660254037844386, 8.6602540378443},
      {26.87936011143123, 19.36491673103708, -3.674234614174766, -19, 7.960841664045326, 19.36491673103708, -10.27740239554723},
      {68.03491750564559, 71.11610225539643, -3.354101966249679, -26.87936011143123, -6.25, 62.45498378832549, -1.118033988749893, -27.19145086235747, -4.437059837324718}
    };

    std::vector<double> h_s = {2.0, 1.0, 1.0};
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao_dx<<<1, 1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), contraction_amplitude_derivatives.data().get(), d_s.data().get());
    run_asserts(data[shell_type]);
  }

  __global__ 
  void call_evaluate_ao_dy(Atomic_Orbital* atomic_orbitals, double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, double x[3]) {
    atomic_orbitals->evaluate_ao_dy(ao_amplitudes, contraction_amplitudes, contraction_amplitude_derivatives, x);
  }

  TEST_P(DeviceSphericalAtomicOrbitalTest, evaluateSphericalAODY) {
    std::vector<std::vector<double>> data = {
      {2},
      {2, 5, 2},
      {8.660254037844386, 8.660254037844386, -5, -3.464101615137754, -8.6602540378443},
      {-10.27740239554723, 19.36491673103708, -7.960841664045326, -19, 3.674234614174766, -19.36491673103708, 26.87936011143123},
      {-68.03491750564559, -27.19145086235747, -3.354101966249679, -62.45498378832549, -6.25, 26.87936011143122, 1.118033988749893, 71.11610225539643, -4.437059837324718}
    };

    std::vector<double> h_s = {1.0, 2.0, 1.0};
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao_dy<<<1, 1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), contraction_amplitude_derivatives.data().get(), d_s.data().get());
    run_asserts(data[shell_type]);
  }

  __global__ 
  void call_evaluate_ao_dz(Atomic_Orbital* atomic_orbitals, double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, double x[3]) {
    atomic_orbitals->evaluate_ao_dz(ao_amplitudes, contraction_amplitudes, contraction_amplitude_derivatives, x);
  }

  TEST_P(DeviceSphericalAtomicOrbitalTest, evaluateSphericalAODZ) {
    std::vector<std::vector<double>> data = {
      {2},
      {2, 2, 5},
      {3.464101615137754, 8.660254037844386, 10, -8.660254037844386, 0},
      {3.162277660168379, 19.36491673103708, 26.94438717061496, 13, -26.94438717061496, 0, 3.162277660168379},
      {0, 20.91650066335189, 76.02631123499285, 64.82669203345178, -5, -64.82669203345179, 0, 20.91650066335189, -5.916079783099618}
    };

    std::vector<double> h_s = {1.0, 1.0, 2.0};
    thrust::device_vector<double> d_s(h_s);
    call_evaluate_ao_dz<<<1, 1>>>(atomic_orbital.data().get(), ao_amplitudes.data().get(), contraction_amplitudes.data().get(), contraction_amplitude_derivatives.data().get(), d_s.data().get());
    run_asserts(data[shell_type]);
  }

  INSTANTIATE_TEST_SUITE_P(InstantiationName,
      DeviceSphericalAtomicOrbitalTest,
      testing::Values(SHELL::S, SHELL::P, SHELL::D, SHELL::F, SHELL::G));
}


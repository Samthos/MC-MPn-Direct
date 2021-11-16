#include "gtest/gtest.h"

#include "../../src/basis/shell.h"

namespace {
  TEST(shellTest, stringToShellType) {
    ASSERT_EQ(SHELL::string_to_shell_type("SP"), SHELL::SP);
    ASSERT_EQ(SHELL::string_to_shell_type("S" ), SHELL::S );
    ASSERT_EQ(SHELL::string_to_shell_type("P" ), SHELL::P );
    ASSERT_EQ(SHELL::string_to_shell_type("D" ), SHELL::D );
    ASSERT_EQ(SHELL::string_to_shell_type("F" ), SHELL::F );
    ASSERT_EQ(SHELL::string_to_shell_type("G" ), SHELL::G );
    ASSERT_EQ(SHELL::string_to_shell_type("H" ), SHELL::H );
    ASSERT_EQ(SHELL::string_to_shell_type("I" ), SHELL::I );
    ASSERT_THROW(SHELL::string_to_shell_type("J" ), std::exception);
  }

  TEST(shellTest, numberOfPolynomialsInt) {
    ASSERT_EQ(4,  SHELL::number_of_polynomials(-1, true));
    ASSERT_EQ(9,  SHELL::number_of_polynomials( 4, true));
    ASSERT_EQ(4,  SHELL::number_of_polynomials(-1, false));
    ASSERT_EQ(15, SHELL::number_of_polynomials( 4, false));
  }

  TEST(shellTest, numberOfPolynomialsShellType) {
    ASSERT_EQ(4,  SHELL::number_of_polynomials(SHELL::SP, true));
    ASSERT_EQ(9,  SHELL::number_of_polynomials(SHELL::G , true));
    ASSERT_EQ(4,  SHELL::number_of_polynomials(SHELL::SP, false));
    ASSERT_EQ(15, SHELL::number_of_polynomials(SHELL::G , false));
  }

  TEST(shellTest, numberOfSphericalPolynomials) {
    ASSERT_EQ(4, SHELL::number_of_spherical_polynomials(SHELL::SP));
    ASSERT_EQ(9, SHELL::number_of_spherical_polynomials(SHELL::G));
  }

  TEST(shellTest, numberOfCartesianPolynomials) {
    ASSERT_EQ(4, SHELL::number_of_cartesian_polynomials(SHELL::SP));
    ASSERT_EQ(15, SHELL::number_of_cartesian_polynomials(SHELL::G));
  }

  SHELL::Contracted_Gaussian make_multicontraction() {
    SHELL::Contracted_Gaussian cg;
    cg.push_back({10.0, {10.0, 10.0}});
    cg.push_back({ 1.0, { 1.0,  1.0}});
    cg.push_back({ 0.1, { 0.1,  0.1}});
    return cg;
  }

  std::vector<double> make_s_result() {
    return {3.821447795835115, 0.0679560193187221, 0.001208447899426908};
  }

  std::vector<double> make_p_result() {
    return {24.60737887501592, 0.1383774604603745, 0.0007781536449176214};
  }

  SHELL::Contracted_Gaussian make_contraction() {
    SHELL::Contracted_Gaussian cg;
    cg.push_back({ 1.0, { 1.0,  1.0}});
    return cg;
  }

  TEST(shellTest, ShellSP) {
    SHELL::Shell shell(SHELL::SP, make_multicontraction());

    auto s_values = make_s_result();
    auto p_values = make_p_result();
    ASSERT_EQ(s_values.size(), shell.contracted_gaussian.size());
    for (int i = 0; i < shell.contracted_gaussian.size(); i++) {
      ASSERT_FLOAT_EQ(s_values[i], shell.contracted_gaussian[i].second[0]);
      ASSERT_FLOAT_EQ(p_values[i], shell.contracted_gaussian[i].second[1]);
    }
  }

  TEST(shellTest, ShellS) {
    SHELL::Shell shell(SHELL::S,  make_multicontraction());

    auto s_values = make_s_result();
    ASSERT_EQ(s_values.size(), shell.contracted_gaussian.size());
    for (int i = 0; i < shell.contracted_gaussian.size(); i++) {
      ASSERT_FLOAT_EQ(s_values[i], shell.contracted_gaussian[i].second[0]);
      ASSERT_FLOAT_EQ(s_values[i], shell.contracted_gaussian[i].second[1]);
    }
  }

  TEST(shellTest, ShellP) {
    SHELL::Shell shell(SHELL::P,  make_multicontraction());
    auto p_values = make_p_result();
    ASSERT_EQ(p_values.size(), shell.contracted_gaussian.size());
    for (int i = 0; i < shell.contracted_gaussian.size(); i++) {
      ASSERT_FLOAT_EQ(p_values[i], shell.contracted_gaussian[i].second[0]);
      ASSERT_FLOAT_EQ(p_values[i], shell.contracted_gaussian[i].second[1]);
    }
  }

  TEST(shellTest, ShellD) {
    SHELL::Shell shell(SHELL::D, make_contraction());
    for (int i = 0; i < shell.contracted_gaussian.size(); i++) {
      ASSERT_FLOAT_EQ(1.64592278064949, shell.contracted_gaussian[i].second[0]);
      ASSERT_FLOAT_EQ(1.64592278064949, shell.contracted_gaussian[i].second[1]);
    }
  }
  
  TEST(shellTest, ShellF) {
    SHELL::Shell shell(SHELL::F, make_contraction());
    for (int i = 0; i < shell.contracted_gaussian.size(); i++) {
      ASSERT_FLOAT_EQ(1.472158089299094, shell.contracted_gaussian[i].second[0]);
      ASSERT_FLOAT_EQ(1.472158089299094, shell.contracted_gaussian[i].second[1]);
    }
  }
  
  TEST(shellTest, ShellG) {
    SHELL::Shell shell(SHELL::G, make_contraction());
    for (int i = 0; i < shell.contracted_gaussian.size(); i++) {
      ASSERT_FLOAT_EQ(1.112846912816406, shell.contracted_gaussian[i].second[0]);
      ASSERT_FLOAT_EQ(1.112846912816406, shell.contracted_gaussian[i].second[1]);
    }
  }

  TEST(shellTest, ShellI) {
    SHELL::Shell shell(SHELL::I, make_contraction());
    for (int i = 0; i < shell.contracted_gaussian.size(); i++) {
      ASSERT_FLOAT_EQ(0.4473812919899841, shell.contracted_gaussian[i].second[0]);
      ASSERT_FLOAT_EQ(0.4473812919899841, shell.contracted_gaussian[i].second[1]);
    }
  }
}

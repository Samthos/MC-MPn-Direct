//
// Created by aedoran on 6/5/18.
#include <thrust/device_vector.h>
#include "gtest/gtest.h"
#include "../../src/ovps_set.h"

namespace {
  template <class T>
  class ovpsSetTest : public testing::Test {
   public:
    void SetUp() override {
      iocc1 = 1;
      iocc2 = 6;
      ivir1 = 6;
      ivir2 = 16;
      lda = ivir2;
      electron_pairs = 10;
      std::vector<double> psi1Tau_prep(lda * electron_pairs);
      std::vector<double> psi2Tau_prep(lda * electron_pairs);

      for (int col = 0, idx = 0; col < electron_pairs; col++) {
        for (int row = 0; row < lda; row++, idx++) {
          psi1Tau_prep[idx] =  idx;
          psi2Tau_prep[idx] = -idx;
        }
      }

      psi1Tau = psi1Tau_prep;
      psi2Tau = psi2Tau_prep;

      ovps_set.resize(electron_pairs);
    }

    void call_check(int sign, int, int, T& array, const char* array_name){}

    void check(int sign, int start, int stop, std::vector<double>& array, const char* array_name) {
      for (int row = 0; row < electron_pairs; row++) {
        for (int col = 0; col < electron_pairs; col++) {
          ASSERT_EQ(array[col * electron_pairs + row], sign * value(row, col, start, stop))
            << "row = " << row << ", col = " << col << " of " << array_name << "\n";
        }
      }
    }

    double value(int row, int col, int start, int stop) {
      if (row == col) {
        return 0;
      }
      return -((-1 + start - stop)*(2*start*start + 3*col*lda*(2*lda*row + start + stop) + start*(-1 + 3*lda*row + 2*stop) + stop*(1 + 3*lda*row + 2*stop)))/6.;
    }

    void fill() {}

    OVPS_Set<T> ovps_set;
    T psi1Tau;
    T psi2Tau;
    int iocc1, iocc2, ivir1, ivir2, lda, electron_pairs;
  };

  template <>
  void ovpsSetTest<std::vector<double>>::call_check(int sign, int start, int stop, std::vector<double>& array, const char* array_name) {
    check(sign, start, stop - 1, array, array_name);
  }

  template<>
  void ovpsSetTest<std::vector<double>>::fill() {
    std::fill(ovps_set.s_11.begin(), ovps_set.s_11.end(), -1.0);
    std::fill(ovps_set.s_12.begin(), ovps_set.s_12.end(), -1.0);
    std::fill(ovps_set.s_21.begin(), ovps_set.s_21.end(), -1.0);
    std::fill(ovps_set.s_22.begin(), ovps_set.s_22.end(), -1.0);
  }

  template <>
  void ovpsSetTest<thrust::device_vector<double>>::call_check(int sign, int start, int stop, thrust::device_vector<double>& array, const char* array_name) {
    std::vector<double> host_array(array.size());
    thrust::copy(array.begin(), array.end(), host_array.begin());
    check(sign, start, stop - 1, host_array, array_name);
  }

  template<>
  void ovpsSetTest<thrust::device_vector<double>>::fill() {
    thrust::fill(ovps_set.s_11.begin(), ovps_set.s_11.end(), -1.0);
    thrust::fill(ovps_set.s_12.begin(), ovps_set.s_12.end(), -1.0);
    thrust::fill(ovps_set.s_21.begin(), ovps_set.s_21.end(), -1.0);
    thrust::fill(ovps_set.s_22.begin(), ovps_set.s_22.end(), -1.0);
  }

  using Implementations = testing::Types<std::vector<double>, thrust::device_vector<double>>;
  TYPED_TEST_SUITE(ovpsSetTest, Implementations);

  TYPED_TEST(ovpsSetTest, Occcupied) {
    this->fill();
    this->ovps_set.update(this->psi1Tau, this->iocc1, 
        this->psi2Tau, this->iocc1, 
        this->iocc2 - this->iocc1, this->lda);
    this->call_check( 1, this->iocc1, this->iocc2, this->ovps_set.s_11, "s_11");
    this->call_check(-1, this->iocc1, this->iocc2, this->ovps_set.s_12, "s_12");
    this->call_check(-1, this->iocc1, this->iocc2, this->ovps_set.s_21, "s_21");
    this->call_check( 1, this->iocc1, this->iocc2, this->ovps_set.s_22, "s_22");
  }

  TYPED_TEST(ovpsSetTest, Virtual) {
    this->fill();
    this->ovps_set.update(this->psi1Tau, this->ivir1, 
        this->psi2Tau, this->ivir1, 
        this->ivir2 - this->ivir1, this->lda);
    this->call_check( 1, this->ivir1, this->ivir2, this->ovps_set.s_11, "s_11");
    this->call_check(-1, this->ivir1, this->ivir2, this->ovps_set.s_12, "s_12");
    this->call_check(-1, this->ivir1, this->ivir2, this->ovps_set.s_21, "s_21");
    this->call_check( 1, this->ivir1, this->ivir2, this->ovps_set.s_22, "s_22");
  }
}

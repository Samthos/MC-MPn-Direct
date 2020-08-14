//
// Created by aedoran on 6/5/18.
#include <thrust/device_vector.h>
#include "gtest/gtest.h"
#include "../../src/ovps_set.h"

#include "ovps_test_helper.h"

namespace {
  std::vector<double> make_psi(int size, double sign) {
    std::vector<double> v(size);
    for (int idx = 0; idx < size; idx++) {
      v[idx] = sign * idx;
    }
    return v;
  }
  
  template <template <class, class> class Container, template <class> class Allocator>
  class ovpsSetFixture {
    public:
      ovpsSetFixture() :
          iocc1(1),
          iocc2(6),
          ivir1(6),
          ivir2(16),
          lda(ivir2),
          electron_pairs(10),
          psi1Tau(make_psi(lda * electron_pairs,  1.0)),
          psi2Tau(make_psi(lda * electron_pairs, -1.0))
      {
        ovps_set.resize(electron_pairs);
      }

      void fill() {
        vfill(ovps_set.s_11);
        vfill(ovps_set.s_12);
        vfill(ovps_set.s_21);
        vfill(ovps_set.s_22);
      }

      void build_occ() {
        fill();
        ovps_set.update(psi1Tau, iocc1, 
          psi2Tau, iocc1, 
          iocc2 - iocc1, lda);
      }

      void build_vir() {
        fill();
        ovps_set.update(psi1Tau, ivir1, 
          psi2Tau, ivir1, 
          ivir2 - ivir1, lda);
      }

      int iocc1, iocc2, ivir1, ivir2, lda, electron_pairs;
      OVPS_Set<Container, Allocator> ovps_set;
      Container<double, Allocator<double>> psi1Tau;
      Container<double, Allocator<double>> psi2Tau;
  };

  template class ovpsSetFixture<std::vector, std::allocator>;
  typedef ovpsSetFixture<std::vector, std::allocator> ovpsSetHostFixture;

  template class ovpsSetFixture<thrust::device_vector, thrust::device_allocator>;
  typedef ovpsSetFixture<thrust::device_vector, thrust::device_allocator> ovpsSetDeviceFixture;

  template <class T>
  class ovpsSetTest : public testing::Test {
   public:
    void check(int sign, int start, int stop, const std::vector<double>& array, const char* array_name) {
      for (int row = 0; row < ovps_set_fixture.electron_pairs; row++) {
        for (int col = 0; col < ovps_set_fixture.electron_pairs; col++) {
          ASSERT_EQ(array[col * ovps_set_fixture.electron_pairs + row], sign * value(row, col, start, stop, ovps_set_fixture.lda))
            << "row = " << row << ", col = " << col << " of " << array_name << "\n";
        }
      }
    }

    double value(int row, int col, int start, int stop, int lda) {
      if (row == col) {
        return 0;
      }
      return -((-1 + start - stop)*(2*start*start + 3*col*lda*(2*lda*row + start + stop) + start*(-1 + 3*lda*row + 2*stop) + stop*(1 + 3*lda*row + 2*stop)))/6.;
    }

    T ovps_set_fixture;
  };

  using Implementations = testing::Types<ovpsSetHostFixture, ovpsSetDeviceFixture>;
  TYPED_TEST_SUITE(ovpsSetTest, Implementations);

  TYPED_TEST(ovpsSetTest, Occcupied) {
    this->ovps_set_fixture.build_occ();

    auto iocc1 = this->ovps_set_fixture.iocc1;
    auto iocc2 = this->ovps_set_fixture.iocc2 - 1;
    this->check( 1, iocc1, iocc2, get_vector(this->ovps_set_fixture.ovps_set.s_11), "s_11");
    this->check(-1, iocc1, iocc2, get_vector(this->ovps_set_fixture.ovps_set.s_12), "s_12");
    this->check(-1, iocc1, iocc2, get_vector(this->ovps_set_fixture.ovps_set.s_21), "s_21");
    this->check( 1, iocc1, iocc2, get_vector(this->ovps_set_fixture.ovps_set.s_22), "s_22");
  }

  TYPED_TEST(ovpsSetTest, Virtual) {
    this->ovps_set_fixture.build_vir();

    auto ivir1 = this->ovps_set_fixture.ivir1;
    auto ivir2 = this->ovps_set_fixture.ivir2 - 1;
    this->check( 1, ivir1, ivir2, get_vector(this->ovps_set_fixture.ovps_set.s_11), "s_11");
    this->check(-1, ivir1, ivir2, get_vector(this->ovps_set_fixture.ovps_set.s_12), "s_12");
    this->check(-1, ivir1, ivir2, get_vector(this->ovps_set_fixture.ovps_set.s_21), "s_21");
    this->check( 1, ivir1, ivir2, get_vector(this->ovps_set_fixture.ovps_set.s_22), "s_22");
  }
}

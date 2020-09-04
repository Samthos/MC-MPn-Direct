//
// Created by aedoran on 6/5/18.
#include <thrust/device_vector.h>
#include <algorithm>
#include <sstream>

#include "gtest/gtest.h"
#include "../test_helper.h"

#include "dummy_tau.h"

#include "../../src/basis/movec_parser.h"
#include "../../src/basis/dummy_movec_parser.h"
#include "../../src/basis/wavefunction.h"
#include "../../src/qc_ovps.h"

namespace {
  template <template <class, class> class Container, template <class> class Allocator>
  class ovpsFixture {
    public:

    ovpsFixture() : 
      electron_pairs(10),
      movecs(new Dummy_Movec_Parser()),
      electron_pair_pos(electron_pairs),
      lda(movecs->ivir2),
      offset(movecs->orbital_energies[0]),
      psi1(&electron_pair_pos, movecs),
      psi2(&electron_pair_pos, movecs)
    {
      std::vector<double> psi = make_psi(electron_pairs, movecs->ivir2, 1.0);
      std::copy(psi.begin(), psi.end(), psi1.psi.begin());
      std::transform(psi.begin(), psi.end(), psi2.psi.begin(), std::negate<>());

      tau = std::shared_ptr<Tau>(new Dummy_Tau(movecs));
      tau->resize(2);

      ovps.init(2, electron_pairs);
    }

    int electron_pairs;

    std::shared_ptr<Movec_Parser> movecs;
    Container<Point, Allocator<Point>> electron_pair_pos;

    int lda;
    double offset;

    std::shared_ptr<Tau> tau;

    Wavefunction<Container, Allocator> psi1;
    Wavefunction<Container, Allocator> psi2;
    OVPS<Container, Allocator> ovps;
  };

  template class ovpsFixture<std::vector, std::allocator>;
  typedef ovpsFixture<std::vector, std::allocator> ovpsHostFixture;

  template class ovpsFixture<thrust::device_vector, thrust::device_allocator>;
  typedef ovpsFixture<thrust::device_vector, thrust::device_allocator> ovpsDeviceFixture;

  template <class T>
  class ovpsTest : public testing::Test {
   public:
    ovpsTest() :
      electron_pairs(ovps_fixture.electron_pairs),
      lda(ovps_fixture.lda),
      offset(ovps_fixture.offset) { }

    std::string array_name(char set, int stop, int start, int index) {
      std::string str;
      std::stringstream ss;
      ss << set << "_set[" << stop << "][" << start << "].s_" << index;
      ss >> str;
      return str;
    }

    void check(int sign, int start, int stop, int n_tau, const std::vector<double>& array, const std::string& array_name) {
      double polygamma_factor = sign * PolyGamma_Difference(start, stop, n_tau+1);
      for (int row = 0; row < electron_pairs; row++) {
        for (int col = 0; col < electron_pairs; col++) {
          ASSERT_FLOAT_EQ(array[col * electron_pairs + row], polygamma_factor* value(row, col))
            << "row = " << row << ", col = " << col << " of " << array_name << "\n";
        }
      }
    }

    double value(int row, int col) {
      if (row == col) {
        return 0;
      }
      return 1.0 / ((row+1) * (col+1));
    }

    T ovps_fixture;
    int electron_pairs;
    int lda;
    double offset;
  };

  using Implementations = testing::Types<ovpsHostFixture, ovpsDeviceFixture>;
  TYPED_TEST_SUITE(ovpsTest, Implementations);

  TYPED_TEST(ovpsTest, update) {
    this->ovps_fixture.ovps.update(this->ovps_fixture.psi1, this->ovps_fixture.psi2, this->ovps_fixture.tau.get());

    for (auto stop = 0; stop < this->ovps_fixture.ovps.o_set.size(); stop++) {
      for (auto start = 0; start < this->ovps_fixture.ovps.o_set[stop].size(); start++) {
        int n_tau = 1 + stop - start;
        this->check( 1, this->ovps_fixture.movecs->iocc1, this->ovps_fixture.movecs->iocc2,  n_tau, get_vector(this->ovps_fixture.ovps.o_set[stop][start].s_11), this->array_name('o', stop, start, 11));
      // this->check(-1, this->ovps_fixture.movecs->iocc1, this->ovps_fixture.movecs->iocc2,  n_tau, get_vector(this->ovps_fixture.ovps.o_set[stop][start].s_12), this->array_name('o', stop, start, 12));
      // this->check(-1, this->ovps_fixture.movecs->iocc1, this->ovps_fixture.movecs->iocc2,  n_tau, get_vector(this->ovps_fixture.ovps.o_set[stop][start].s_21), this->array_name('o', stop, start, 21));
      // this->check( 1, this->ovps_fixture.movecs->iocc1, this->ovps_fixture.movecs->iocc2,  n_tau, get_vector(this->ovps_fixture.ovps.o_set[stop][start].s_22), this->array_name('o', stop, start, 22));
      //                 
      // this->check( 1, this->ovps_fixture.movecs->ivir1, this->ovps_fixture.movecs->ivir2, -n_tau, get_vector(this->ovps_fixture.ovps.v_set[stop][start].s_11), this->array_name('v', stop, start, 11));
      // this->check(-1, this->ovps_fixture.movecs->ivir1, this->ovps_fixture.movecs->ivir2, -n_tau, get_vector(this->ovps_fixture.ovps.v_set[stop][start].s_12), this->array_name('v', stop, start, 12));
      // this->check(-1, this->ovps_fixture.movecs->ivir1, this->ovps_fixture.movecs->ivir2, -n_tau, get_vector(this->ovps_fixture.ovps.v_set[stop][start].s_21), this->array_name('v', stop, start, 21));
      // this->check( 1, this->ovps_fixture.movecs->ivir1, this->ovps_fixture.movecs->ivir2, -n_tau, get_vector(this->ovps_fixture.ovps.v_set[stop][start].s_22), this->array_name('v', stop, start, 22));
      }
    }
  }
}

//
// Created by aedoran on 6/5/18.
#include <thrust/device_vector.h>
#include <algorithm>
#include <sstream>
#include "gtest/gtest.h"

#include "stochastic_tau.h"

#include "../../src/basis/movec_parser.h"
#include "../../src/basis/dummy_movec_parser.h"
#include "../../src/basis/wavefunction.h"
#include "../../src/qc_ovps.h"

#include "ovps_test_helper.h"


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
      std::iota(psi1.psi.begin(), psi1.psi.end(), 0.0);
      std::iota(psi2.psi.begin(), psi2.psi.end(), 0.0);
      std::transform(psi2.psi.begin(), psi2.psi.end(), psi2.psi.begin(), std::negate<>());

      std::shared_ptr<Stochastic_Tau> stochastic_tau(new Stochastic_Tau(movecs));
      stochastic_tau->resize(2);
      stochastic_tau->set({1.0-1.0/exp(2), 1.0-1.0/exp(2)});
      tau = std::shared_ptr<Tau>(stochastic_tau);

      ovps.init(2, electron_pairs);
    }

    int electron_pairs;

    std::shared_ptr<Movec_Parser> movecs;
    std::vector<Point> electron_pair_pos;

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
      stop -= 1;
      for (int row = 0; row < electron_pairs; row++) {
        for (int col = 0; col < electron_pairs; col++) {
          ASSERT_FLOAT_EQ(array[col * electron_pairs + row], sign * value(row, col, start, stop, n_tau))
            << "row = " << row << ", col = " << col << " of " << array_name << "\n";
        }
      }
    }

    double value(int row, int col, int start, int stop, int n) {
      if (row == col) {
        return 0;
      }
      return (
       - exp(n*(start + 0))*(col*lda + start)*(lda*row + start)
       + exp(n*(start + 1))*(-1 + 2*(-1 + start)*start + lda*row*(-1 + 2*start) + col*lda*(-1 + 2*lda*row + 2*start))
       - exp(n*(start + 2))*(col*lda + start - 1)*(lda*row + start - 1)
       + exp(n*(stop + 1))*(col*lda + stop + 1)*(lda*row + stop + 1) 
       - exp(n*(stop + 2))*(-1 + lda*(col + row + 2*col*lda*row) + 2*stop + 2*lda*(col + row)*stop + 2*stop*stop)
       + exp(n*(stop + 3))*(col*lda + stop)*(lda*row + stop)
       ) / (exp(-n*offset) * pow(exp(n) - 1,3));
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
        this->check(-1, this->ovps_fixture.movecs->iocc1, this->ovps_fixture.movecs->iocc2,  n_tau, get_vector(this->ovps_fixture.ovps.o_set[stop][start].s_12), this->array_name('o', stop, start, 12));
        this->check(-1, this->ovps_fixture.movecs->iocc1, this->ovps_fixture.movecs->iocc2,  n_tau, get_vector(this->ovps_fixture.ovps.o_set[stop][start].s_21), this->array_name('o', stop, start, 21));
        this->check( 1, this->ovps_fixture.movecs->iocc1, this->ovps_fixture.movecs->iocc2,  n_tau, get_vector(this->ovps_fixture.ovps.o_set[stop][start].s_22), this->array_name('o', stop, start, 22));
                        
        this->check( 1, this->ovps_fixture.movecs->ivir1, this->ovps_fixture.movecs->ivir2, -n_tau, get_vector(this->ovps_fixture.ovps.v_set[stop][start].s_11), this->array_name('v', stop, start, 11));
        this->check(-1, this->ovps_fixture.movecs->ivir1, this->ovps_fixture.movecs->ivir2, -n_tau, get_vector(this->ovps_fixture.ovps.v_set[stop][start].s_12), this->array_name('v', stop, start, 12));
        this->check(-1, this->ovps_fixture.movecs->ivir1, this->ovps_fixture.movecs->ivir2, -n_tau, get_vector(this->ovps_fixture.ovps.v_set[stop][start].s_21), this->array_name('v', stop, start, 21));
        this->check( 1, this->ovps_fixture.movecs->ivir1, this->ovps_fixture.movecs->ivir2, -n_tau, get_vector(this->ovps_fixture.ovps.v_set[stop][start].s_22), this->array_name('v', stop, start, 22));
      }
    }
  }
}

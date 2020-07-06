//
// Created by aedoran on 6/5/18.
#include <thrust/device_vector.h>
#include <sstream>
#include "gtest/gtest.h"

#include "stochastic_tau.h"

#include "../../src/basis/movec_parser.h"
#include "../../src/basis/dummy_movec_parser.h"
#include "../../src/basis/wavefunction.h"
#include "../../src/qc_ovps.h"


namespace {
  template <class T>
  class ovpsTest : public testing::Test {
   public:
    void SetUp() override {
      electron_pairs = 10;
      std::vector<std::array<double, 3>> electron_pair_pos(electron_pairs);
      movecs = std::shared_ptr<Movec_Parser>(new Dummy_Movec_Parser());

      lda = movecs->ivir2;
      offset = movecs->orbital_energies[0];

      psi1 = Wavefunction(&electron_pair_pos, movecs);
      psi2 = Wavefunction(&electron_pair_pos, movecs);
      std::iota(psi1.psi.begin(), psi1.psi.end(), 0.0);
      std::iota(psi2.psi.begin(), psi2.psi.end(), 0.0);
      std::transform(psi2.psi.begin(), psi2.psi.end(), psi2.psi.begin(), std::negate<>());

      std::shared_ptr<Stochastic_Tau> stochastic_tau(new Stochastic_Tau(movecs));
      stochastic_tau->resize(2);
      stochastic_tau->set({1.0-1.0/exp(2), 1.0-1.0/exp(2)});

      tau = std::shared_ptr<Tau>(stochastic_tau);

      ovps.init(2, electron_pairs);
    }

    std::string array_name(char set, int stop, int start, int index) {
      std::string str;
      std::stringstream ss;
      ss << set << "_set[" << stop << "][" << start << "].s_" << index;
      ss >> str;
      return str;
    }

    void call_check(int sign, int, int, int, T& array, const std::string& array_name){}

    void check(int sign, int start, int stop, int n_tau, std::vector<double>& array, const std::string& array_name) {
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

    int electron_pairs;
    std::shared_ptr<Movec_Parser> movecs;
    int lda;
    double offset;
    Wavefunction psi1;
    Wavefunction psi2;
    std::shared_ptr<Tau> tau;
    OVPS<T> ovps;
  };

  template <>
  void ovpsTest<std::vector<double>>::call_check(int sign, int start, int stop, int n_tau, std::vector<double>& array, const std::string& array_name) {
    check(sign, start, stop - 1, n_tau, array, array_name);
  }

  template <>
  void ovpsTest<thrust::device_vector<double>>::call_check(int sign, int start, int stop, int n_tau, thrust::device_vector<double>& array, const std::string& array_name) {
    std::vector<double> host_array(array.size());
    thrust::copy(array.begin(), array.end(), host_array.begin());
    check(sign, start, stop - 1, n_tau, host_array, array_name);
  }

  using Implementations = testing::Types<std::vector<double>, thrust::device_vector<double>>;
  TYPED_TEST_SUITE(ovpsTest, Implementations);

  TYPED_TEST(ovpsTest, update) {
    this->ovps.update(this->psi1, this->psi2, this->tau.get());

    for (auto stop = 0; stop < this->ovps.o_set.size(); stop++) {
      for (auto start = 0; start < this->ovps.o_set[stop].size(); start++) {
        int n_tau = 1 + stop - start;
        this->call_check(1, this->movecs->iocc1, this->movecs->iocc2,  n_tau, this->ovps.o_set[stop][start].s_11, this->array_name('o', stop, start, 11));
        this->call_check(-1, this->movecs->iocc1, this->movecs->iocc2,  n_tau, this->ovps.o_set[stop][start].s_12, this->array_name('o', stop, start, 12));
        this->call_check(-1, this->movecs->iocc1, this->movecs->iocc2,  n_tau, this->ovps.o_set[stop][start].s_21, this->array_name('o', stop, start, 21));
        this->call_check(1, this->movecs->iocc1, this->movecs->iocc2,  n_tau, this->ovps.o_set[stop][start].s_22, this->array_name('o', stop, start, 22));

        this->call_check(1, this->movecs->ivir1, this->movecs->ivir2, -n_tau, this->ovps.v_set[stop][start].s_11, this->array_name('v', stop, start, 11));
        this->call_check(-1, this->movecs->ivir1, this->movecs->ivir2, -n_tau, this->ovps.v_set[stop][start].s_12, this->array_name('v', stop, start, 12));
        this->call_check(-1, this->movecs->ivir1, this->movecs->ivir2, -n_tau, this->ovps.v_set[stop][start].s_21, this->array_name('v', stop, start, 21));
        this->call_check(1, this->movecs->ivir1, this->movecs->ivir2, -n_tau, this->ovps.v_set[stop][start].s_22, this->array_name('v', stop, start, 22));
      }
    }
  }
}

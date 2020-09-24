//
// Created by aedoran on 6/5/18.
#include <algorithm>

#include <list>

#include "gtest/gtest.h"
#include "../test_helper.h"

#include "dummy_tau.h"

#include "../../src/basis/movec_parser.h"
#include "../../src/basis/dummy_movec_parser.h"
#include "../../src/basis/wavefunction.h"
#include "../../src/electron_generators/dummy_electron_pair_list.h"
#include "../../src/qc_ovps.h"

#include "../../src/MCMP/create_mp2_functional.h"

namespace {
  template <template <typename, typename> typename Container, template <typename> typename Allocator>
  class MP_Functional_Fixture {
    public:
      MP_Functional_Fixture(int n_tau) :
        electron_pairs(10),
        movecs(new Dummy_Movec_Parser()),
        lda(movecs->ivir2),
        offset(movecs->orbital_energies[0]),
        electron_pair_list(new Dummy_Electron_Pair_List<Container, Allocator>(electron_pairs)),
        psi1(&electron_pair_list->pos1, movecs),
        psi2(&electron_pair_list->pos2, movecs)
      {
        std::vector<double> psi = make_psi(electron_pairs, movecs->ivir2, 1.0);
        std::copy(psi.begin(), psi.end(), psi1.psi.begin());
        std::transform(psi.begin(), psi.end(), psi2.psi.begin(), std::negate<>());

        tau = std::shared_ptr<Tau>(new Dummy_Tau(movecs));
        tau->resize(1);

        ovps.init(1, electron_pairs);
        ovps.update(psi1, psi2, tau.get());
      }

      int electron_pairs;

      std::shared_ptr<Movec_Parser> movecs;
      std::shared_ptr<Electron_Pair_List<Container, Allocator>> electron_pair_list;

      int lda;
      double offset;

      std::shared_ptr<Tau> tau;
      Wavefunction<Container, Allocator> psi1;
      Wavefunction<Container, Allocator> psi2;
      OVPS<Container, Allocator> ovps;
  };

  template <template <typename, typename> typename Container, template <typename> typename Allocator, bool standard, int CV> 
  class MP2_Functional_Fixture : public MP_Functional_Fixture<Container, Allocator> {
    typedef Standard_MP_Functional<Container, Allocator> Standard_Functional_Type;
    typedef Direct_MP_Functional<Container, Allocator> Direct_Functional_Type;

    public:
      MP2_Functional_Fixture() : MP_Functional_Fixture<Container, Allocator>(1) {
        if (standard) {
          mp_functional = create_MP2_Functional<Container, Allocator>(CV, this->electron_pairs);
        } else {
          mp_functional = create_Direct_MP2_Functional(CV);
        }
        control.resize(mp_functional->n_control_variates);
      }

      void energy() {
        emp = 0;
        std::fill(control.begin(), control.end(), 0);
        if (mp_functional->functional_type == MP_FUNCTIONAL_TYPE::STANDARD) {
          Standard_Functional_Type* functional = dynamic_cast<Standard_Functional_Type*>(mp_functional);
          functional->energy(emp, control, this->ovps, this->electron_pair_list.get(), this->tau.get());
        } else if (mp_functional->functional_type == MP_FUNCTIONAL_TYPE::DIRECT) {
          Direct_Functional_Type* functional = dynamic_cast<Direct_Functional_Type*>(mp_functional);
          functional->energy(emp, control, this->psi1, this->psi2, this->electron_pair_list.get(), this->tau.get());
        }
      }

      double emp;
      std::vector<double> control;
      MP_Functional* mp_functional;
  };

  template <typename T>
  class MP2FunctionalTest : public testing::Test {
   public:
     T mp_functional_fixture;
  };

  using Implementations = testing::Types<
    MP2_Functional_Fixture<std::vector, std::allocator, true, 0>,
    MP2_Functional_Fixture<std::vector, std::allocator, true, 1>,
    MP2_Functional_Fixture<std::vector, std::allocator, true, 2>,
    MP2_Functional_Fixture<thrust::device_vector, thrust::device_allocator, true, 0>,
    MP2_Functional_Fixture<thrust::device_vector, thrust::device_allocator, true, 1>,
    MP2_Functional_Fixture<thrust::device_vector, thrust::device_allocator, true, 2>,
    MP2_Functional_Fixture<std::vector, std::allocator, false, 0>,
    MP2_Functional_Fixture<std::vector, std::allocator, false, 1>,
    MP2_Functional_Fixture<std::vector, std::allocator, false, 2>
    >;
  TYPED_TEST_SUITE(MP2FunctionalTest, Implementations);

  TYPED_TEST(MP2FunctionalTest, EnergyTest) {
    this->mp_functional_fixture.energy();

    double emp;
    emp  = PolyGamma_Difference(0, 10, 3);
    emp *= emp;
    emp -= PolyGamma_Difference(0, 10, 7);
    emp *= PolyGamma_Difference(this->mp_functional_fixture.movecs->iocc1, 
        this->mp_functional_fixture.movecs->iocc2, 2);
    emp *= PolyGamma_Difference(this->mp_functional_fixture.movecs->iocc1, 
        this->mp_functional_fixture.movecs->iocc2, 2);
    emp *= PolyGamma_Difference(this->mp_functional_fixture.movecs->ivir1, 
        this->mp_functional_fixture.movecs->ivir2, 2);
    emp *= PolyGamma_Difference(this->mp_functional_fixture.movecs->ivir1, 
        this->mp_functional_fixture.movecs->ivir2, 2);
    emp *= 2.0; // tau weight
    emp /= 10.0 * 9.0; // electron pairs;
    emp *= -1;


    ASSERT_FLOAT_EQ(this->mp_functional_fixture.emp, emp);
    for (auto &it : this->mp_functional_fixture.control) {
      ASSERT_FLOAT_EQ(it, -emp);
    }
  }

}

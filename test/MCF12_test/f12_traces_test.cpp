#include <unordered_map>
#include "gtest/gtest.h"
#include "test_helper.h"

#include "dummy_movec_parser.h"
#include "dummy_electron_list.h"
#include "dummy_electron_pair_list.h"
#include "wavefunction.h"
#include "../../src/MCF12/F12_Traces.h"

namespace {
  template <template <typename, typename> typename Container, template <typename> typename Allocator>
  class F12_Traces_Fixture {
    typedef Container<double, Allocator<double>> vector_double;
    typedef Container<Point, Allocator<Point>> vector_Point;

    typedef Dummy_Electron_List<Container, Allocator> Electron_List_Type;
    typedef Dummy_Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
    typedef Wavefunction<Container, Allocator> Wavefunction_Type;
    typedef F12_Traces<Container, Allocator> F12_Traces_Type;

    public:
      F12_Traces_Fixture() :
          electrons(16),
          electron_pairs(12),
          electron_list(new Electron_List_Type(electrons)),
          electron_pair_list(new Electron_Pair_List_Type(electron_pairs)),
          movecs(new Dummy_Movec_Parser()),
          f12_traces(electron_pairs, electrons)
      {
        emplace(WC::electrons, &electron_list->pos,    -1.0);
     // emplace(WC::electrons_dx, &electron_list->pos, -2.0);
     // emplace(WC::electrons_dy, &electron_list->pos, -3.0);
     // emplace(WC::electrons_dz, &electron_list->pos, -4.0);

        emplace(WC::electron_pairs_1, &electron_pair_list->pos1,    1.0);
     // emplace(WC::electron_pairs_1_dx, &electron_pair_list->pos1, 2.0);
     // emplace(WC::electron_pairs_1_dy, &electron_pair_list->pos1, 3.0);
     // emplace(WC::electron_pairs_1_dz, &electron_pair_list->pos1, 4.0);

        emplace(WC::electron_pairs_2, &electron_pair_list->pos2,    10.0);
      // emplace(WC::electron_pairs_2_dx, &electron_pair_list->pos2, 20.0);
      // emplace(WC::electron_pairs_2_dy, &electron_pair_list->pos2, 30.0);
      // emplace(WC::electron_pairs_2_dz, &electron_pair_list->pos2, 40.0);
        f12_traces.update_v(wavefunctions);
      }

      void emplace(WC::Wavefunction_Code code, vector_Point* pos, double sign) {
        auto v = make_psi(pos->size(), movecs->ivir2, sign);
        wavefunctions.emplace(code, Wavefunction_Type(pos, movecs));
        wavefunction_multipliers.emplace(code, sign);
#ifdef HAVECUDA
        namespace NS = thrust;
#else
        namespace NS = std;
#endif
        NS::copy(v.begin(), v.end(), wavefunctions[code].psi.begin());
      }

      int electrons;
      int electron_pairs;
      std::shared_ptr<Electron_List_Type> electron_list;
      std::shared_ptr<Electron_Pair_List_Type> electron_pair_list;
      std::shared_ptr<Movec_Parser> movecs;
      std::unordered_map<int, Wavefunction_Type> wavefunctions;
      std::unordered_map<int, double> wavefunction_multipliers;
      F12_Traces_Type f12_traces;
  };

  template <typename T>
  class F12TracesTest : public testing::Test {
    public:
      void SetUp() {
        iocc1 = this->f12_traces_fixture.movecs->iocc1;
        iocc2 = this->f12_traces_fixture.movecs->iocc2;
        ivir1 = this->f12_traces_fixture.movecs->ivir1;
        ivir2 = this->f12_traces_fixture.movecs->ivir2;
        electrons = this->f12_traces_fixture.electron_list->size();
        electron_pairs = this->f12_traces_fixture.electron_pair_list->size();
      }

      void m_check(int m, int n, double ref, const std::vector<double>& v, std::string name, bool diag_is_zero) {
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            auto value = ref / (i + 1) / (j + 1);
            if (i == j && diag_is_zero) {
              value = 0.0;
            }
            ASSERT_FLOAT_EQ(v[i * n + j], value) << name << "["  << i << ", " << j << "]";
          }
        }
      }

      void v_check(int m, double ref, const std::vector<double>& v, std::string name) {
        for (int i = 0; i < m; i++) {
          auto value = ref / (i + 1) / (i + 1);
          ASSERT_FLOAT_EQ(v[i], value) << name << "["  << i << "]";
        }
      }

      int iocc1;
      int iocc2;
      int ivir1;
      int ivir2;
      int electrons;
      int electron_pairs;
      T f12_traces_fixture;
  };

#ifdef HAVE_CUDA
  using Implementations = testing::Types<
    F12_Traces_Fixture<std::vector, std::allocator>,
    F12_Traces_Fixture<thrust::device_vector, thrust::device_allocator>
    >;
#else 
  using Implementations = testing::Types<
    F12_Traces_Fixture<std::vector, std::allocator>
    >;
#endif
  TYPED_TEST_SUITE(F12TracesTest, Implementations);

  TYPED_TEST(F12TracesTest, op11Test) { 
    auto& wm = this->f12_traces_fixture.wavefunction_multipliers;
    this->v_check(
        this->electron_pairs,
        PolyGamma_Difference(this->iocc1, this->iocc2, 1) * wm[WC::electrons] * wm[WC::electrons],
        get_vector(this->f12_traces_fixture.f12_traces.op11),
        "op11");
  }

  TYPED_TEST(F12TracesTest, ElectronElectronTest) { 
    auto& wm = this->f12_traces_fixture.wavefunction_multipliers;
    this->m_check(
        this->electrons, this->electrons,
        PolyGamma_Difference(this->iocc1, this->iocc2, 1) * wm[WC::electrons] * wm[WC::electrons],
        get_vector(this->f12_traces_fixture.f12_traces.op12),
        "op12", true);
    this->m_check(
        this->electrons, this->electrons,
        PolyGamma_Difference(this->ivir1, this->ivir2, 1) * wm[WC::electrons] * wm[WC::electrons],
        get_vector(this->f12_traces_fixture.f12_traces.ov12),
        "ov12", true);
    this->m_check(
        this->electrons, this->electrons,
        PolyGamma_Difference(0, this->iocc2, 1) * wm[WC::electrons] * wm[WC::electrons],
        get_vector(this->f12_traces_fixture.f12_traces.ok12),
        "ok12", true);
  }

  TYPED_TEST(F12TracesTest, ElectronPairTest) { 
    auto& wm = this->f12_traces_fixture.wavefunction_multipliers;
    this->v_check(
        this->electron_pairs,
        PolyGamma_Difference(this->iocc1, this->iocc2, 1) * wm[WC::electron_pairs_1] * wm[WC::electron_pairs_1],
        get_vector(this->f12_traces_fixture.f12_traces.p11),
        "p11");
    this->v_check(
        this->electron_pairs,
        PolyGamma_Difference(this->iocc1, this->iocc2, 1) * wm[WC::electron_pairs_1] * wm[WC::electron_pairs_2],
        get_vector(this->f12_traces_fixture.f12_traces.p12),
        "p12");
    this->v_check(
        this->electron_pairs,
        PolyGamma_Difference(this->iocc1, this->iocc2, 1) * wm[WC::electron_pairs_2] * wm[WC::electron_pairs_2],
        get_vector(this->f12_traces_fixture.f12_traces.p22),
        "p22");
    this->v_check(
        this->electron_pairs,
        PolyGamma_Difference(0, this->iocc2, 1) * wm[WC::electron_pairs_1] * wm[WC::electron_pairs_2],
        get_vector(this->f12_traces_fixture.f12_traces.k12),
        "k12");
  }

  TYPED_TEST(F12TracesTest, ElectronPairElectronTest) {
    auto& wm = this->f12_traces_fixture.wavefunction_multipliers;
    this->m_check(
        this->electron_pairs, this->electrons,
        PolyGamma_Difference(this->iocc1, this->iocc2, 1) * wm[WC::electrons] * wm[WC::electron_pairs_1],
        get_vector(this->f12_traces_fixture.f12_traces.p13),
        "p13", false);
    this->m_check(
        this->electron_pairs, this->electrons,
        PolyGamma_Difference(this->iocc1, this->iocc2, 1) * wm[WC::electrons] * wm[WC::electron_pairs_2],
        get_vector(this->f12_traces_fixture.f12_traces.p23),
        "p23", false);
    this->m_check(
        this->electron_pairs, this->electrons,
        PolyGamma_Difference(this->ivir1, this->ivir2, 1) * wm[WC::electrons] * wm[WC::electron_pairs_1],
        get_vector(this->f12_traces_fixture.f12_traces.v13),
        "v13", false);
    this->m_check(
        this->electron_pairs, this->electrons,
        PolyGamma_Difference(this->ivir1, this->ivir2, 1) * wm[WC::electrons] * wm[WC::electron_pairs_2],
        get_vector(this->f12_traces_fixture.f12_traces.v23),
        "v23", false);
    this->m_check(
        this->electron_pairs, this->electrons,
        PolyGamma_Difference(0, this->iocc2, 1) * wm[WC::electrons] * wm[WC::electron_pairs_1],
        get_vector(this->f12_traces_fixture.f12_traces.k13),
        "k13", false);
    this->m_check(
        this->electron_pairs, this->electrons,
        PolyGamma_Difference(0, this->iocc2, 1) * wm[WC::electrons] * wm[WC::electron_pairs_2],
        get_vector(this->f12_traces_fixture.f12_traces.k23),
        "k23", false);
  }
}

#include "../../src/MCF12/correlation_factor_data.h"
#include "dummy_electron_list.h"
#include "dummy_electron_pair_list.h"
#include "gtest/gtest.h"
#include "test_helper.h"

namespace {
std::vector<double> f12p_result() {
  std::vector<double> result = {
      0.0000000000000000e+00, 8.9125887066303333e-01, 1.0228390753523038e+00, 1.0757796642511721e+00,
      1.1043596489127128e+00, 1.1222483533326908e+00, 1.1344996331410484e+00, 1.1434156023585165e+00,
      1.1501950915696266e+00, 1.1555238656732347e+00};
  return result;
}
std::vector<double> f12pa_result() {
  std::vector<double> result = {
      -2.0000000000000000e+00, -3.4061966915876672e-02, -6.4356152548617736e-03, -2.2185251364769216e-03,
      -1.0125342967311672e-03, -5.4402022294892027e-04, -3.2525011200960349e-04, -2.0968933802809997e-04,
      -1.4298902188721613e-04, -1.0182803359465335e-04};
  return result;
}
std::vector<double> f12pc_result() {
  std::vector<double> result = {
      1.0000000000000000e+00, 2.5728427444747209e-01, 1.4763410387308015e-01, 1.0351694645735657e-01,
      7.9700292572739417e-02, 6.4793038889424323e-02, 5.4583639049126331e-02, 4.7153664701236254e-02,
      4.1504090358644392e-02, 3.7063445272304388e-02};
  return result;
}
std::vector<double> f12o_result() {
  std::vector<double> result = {
      0.0000000000000000e+00, 1.5135838034745985e-01, 2.6881108528775505e-01, 3.6260337510235469e-01,
      4.3923048454132646e-01, 5.0300953833274153e-01, 5.5692193816530555e-01, 6.0309286568572307e-01,
      1.5135838034745985e-01, 0.0000000000000000e+00, 1.5135838034745985e-01, 2.6881108528775505e-01,
      3.6260337510235469e-01, 4.3923048454132646e-01, 5.0300953833274165e-01, 5.5692193816530555e-01,
      2.6881108528775505e-01, 1.5135838034745985e-01, 0.0000000000000000e+00, 1.5135838034745983e-01,
      2.6881108528775505e-01, 3.6260337510235469e-01, 4.3923048454132646e-01, 5.0300953833274153e-01,
      3.6260337510235469e-01, 2.6881108528775505e-01, 1.5135838034745983e-01, 0.0000000000000000e+00,
      1.5135838034745977e-01, 2.6881108528775499e-01, 3.6260337510235469e-01, 4.3923048454132646e-01,
      4.3923048454132646e-01, 3.6260337510235469e-01, 2.6881108528775505e-01, 1.5135838034745977e-01,
      0.0000000000000000e+00, 1.5135838034745977e-01, 2.6881108528775510e-01, 3.6260337510235469e-01,
      5.0300953833274153e-01, 4.3923048454132646e-01, 3.6260337510235469e-01, 2.6881108528775499e-01,
      1.5135838034745977e-01, 0.0000000000000000e+00, 1.5135838034745994e-01, 2.6881108528775510e-01,
      5.5692193816530555e-01, 5.0300953833274165e-01, 4.3923048454132646e-01, 3.6260337510235469e-01,
      2.6881108528775510e-01, 1.5135838034745994e-01, 0.0000000000000000e+00, 1.5135838034745977e-01,
      6.0309286568572307e-01, 5.5692193816530555e-01, 5.0300953833274153e-01, 4.3923048454132646e-01,
      3.6260337510235469e-01, 2.6881108528775510e-01, 1.5135838034745977e-01, 0.0000000000000000e+00};
  return result;
}
std::vector<double> f12ob_result() {
  std::vector<double> result = {
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
      0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00};
  return result;
}
std::vector<double> f12od_result() {
  std::vector<double> result = {
      0.0000000000000000e+00, -6.3637109170573081e-01, -5.0180138592764378e-01, -4.0580619640623128e-01,
      -3.3493649053890340e-01, -2.8113177294857528e-01, -2.3932256574830280e-01, -2.0619104571486252e-01,
      -6.3637109170573081e-01, 0.0000000000000000e+00, -6.3637109170573081e-01, -5.0180138592764378e-01,
      -4.0580619640623128e-01, -3.3493649053890340e-01, -2.8113177294857528e-01, -2.3932256574830280e-01,
      -5.0180138592764378e-01, -6.3637109170573081e-01, 0.0000000000000000e+00, -6.3637109170573058e-01,
      -5.0180138592764378e-01, -4.0580619640623128e-01, -3.3493649053890340e-01, -2.8113177294857528e-01,
      -4.0580619640623128e-01, -5.0180138592764378e-01, -6.3637109170573058e-01, 0.0000000000000000e+00,
      -6.3637109170573081e-01, -5.0180138592764389e-01, -4.0580619640623128e-01, -3.3493649053890340e-01,
      -3.3493649053890340e-01, -4.0580619640623128e-01, -5.0180138592764378e-01, -6.3637109170573081e-01,
      0.0000000000000000e+00, -6.3637109170573081e-01, -5.0180138592764378e-01, -4.0580619640623128e-01,
      -2.8113177294857528e-01, -3.3493649053890340e-01, -4.0580619640623128e-01, -5.0180138592764389e-01,
      -6.3637109170573081e-01, 0.0000000000000000e+00, -6.3637109170573058e-01, -5.0180138592764378e-01,
      -2.3932256574830280e-01, -2.8113177294857528e-01, -3.3493649053890340e-01, -4.0580619640623128e-01,
      -5.0180138592764378e-01, -6.3637109170573058e-01, 0.0000000000000000e+00, -6.3637109170573081e-01,
      -2.0619104571486252e-01, -2.3932256574830280e-01, -2.8113177294857528e-01, -3.3493649053890340e-01,
      -4.0580619640623128e-01, -5.0180138592764378e-01, -6.3637109170573081e-01, 0.0000000000000000e+00};
  return result;
}
std::vector<double> f13_result() {
  std::vector<double> result = {
      0.0000000000000000e+00, 1.5135838034745985e-01, 2.6881108528775505e-01, 3.6260337510235469e-01,
      4.3923048454132646e-01, 5.0300953833274153e-01, 5.5692193816530555e-01, 6.0309286568572307e-01,
      7.0887617762872868e-01, 6.7804257918256072e-01, 6.4307806183469451e-01, 6.0309286568572285e-01,
      5.5692193816530555e-01, 5.0300953833274153e-01, 4.3923048454132624e-01, 3.6260337510235463e-01,
      8.9125887066303333e-01, 8.7935133019079348e-01, 8.6648844284800353e-01, 8.5255043196008140e-01,
      8.3739662489764544e-01, 8.2086068844655835e-01, 8.0274449818494220e-01, 7.8281016204712450e-01,
      9.7486465224187979e-01, 9.6859839544314275e-01, 9.6197333079177005e-01, 9.5495773177543830e-01,
      9.4751601785892370e-01, 9.3960815093888161e-01, 9.3118891471224485e-01, 9.2220704957254862e-01,
      1.0228390753523038e+00, 1.0189817303619186e+00, 1.0149526738534647e+00, 1.0107401790883932e+00,
      1.0063314266443093e+00, 1.0017123741108145e+00, 9.9686760668299557e-01, 9.9178016530364044e-01,
      1.0539591379214803e+00, 1.0513479175774756e+00, 1.0486416196525401e+00, 1.0458349550265902e+00,
      1.0429222348591836e+00, 1.0398973321017313e+00, 1.0367536384753460e+00, 1.0334840162786141e+00,
      1.0757796642511721e+00, 1.0738954882891041e+00, 1.0719532735744344e+00, 1.0699502964852092e+00,
      1.0678836602737971e+00, 1.0657502810889543e+00, 1.0635468726213901e+00, 1.0612699292120529e+00,
      1.0919272317924273e+00, 1.0905038770492654e+00, 1.0890425298097723e+00, 1.0875416483417810e+00,
      1.0859996063514807e+00, 1.0844146871052216e+00, 1.0827850770540615e+00, 1.0811088589112909e+00,
      1.1043596489127128e+00, 1.1032466226013051e+00, 1.1021073852959267e+00, 1.1009410000888527e+00,
      1.0997464848810883e+00, 1.0985228096243866e+00, 1.0972688933587942e+00, 1.0959836010278101e+00,
      1.1142267823062899e+00, 1.1133326425767274e+00, 1.1124196646157622e+00, 1.1114872467436825e+00,
      1.1105347613820860e+00, 1.1095615536452414e+00, 1.1085669398385030e+00, 1.1075502058565354e+00};
  return result;
}
std::vector<double> f23_result() {
  std::vector<double> result = {
      0.0000000000000000e+00, 1.5135838034745985e-01, 2.6881108528775505e-01, 3.6260337510235469e-01,
      4.3923048454132646e-01, 5.0300953833274153e-01, 5.5692193816530555e-01, 6.0309286568572307e-01,
      7.0887617762872868e-01, 7.3627010082689426e-01, 7.6076951545867355e-01, 7.8281016204712439e-01,
      8.0274449818494220e-01, 8.2086068844655835e-01, 8.3739662489764544e-01, 8.5255043196008151e-01,
      8.9125887066303333e-01, 9.0231368599760087e-01, 9.1260420807506093e-01, 9.2220704957254862e-01,
      9.3118891471224485e-01, 9.3960815093888161e-01, 9.4751601785892370e-01, 9.5495773177543830e-01,
      9.7486465224187979e-01, 9.8080048174561996e-01, 9.8643134831702917e-01, 9.9178016530364044e-01,
      9.9686760668299557e-01, 1.0017123741108145e+00, 1.0063314266443093e+00, 1.0107401790883932e+00,
      1.0228390753523038e+00, 1.0265354568710623e+00, 1.0300807442856144e+00, 1.0334840162786141e+00,
      1.0367536384753460e+00, 1.0398973321017313e+00, 1.0429222348591836e+00, 1.0458349550265902e+00,
      1.0539591379214803e+00, 1.0564802045856401e+00, 1.0589157072405151e+00, 1.0612699292120529e+00,
      1.0635468726213901e+00, 1.0657502810889543e+00, 1.0678836602737971e+00, 1.0699502964852092e+00,
      1.0757796642511721e+00, 1.0776083647001995e+00, 1.0793840041273715e+00, 1.0811088589112909e+00,
      1.0827850770540615e+00, 1.0844146871052216e+00, 1.0859996063514807e+00, 1.0875416483417810e+00,
      1.0919272317924273e+00, 1.0933140566353949e+00, 1.0946657400514401e+00, 1.0959836010278101e+00,
      1.0972688933587942e+00, 1.0985228096243866e+00, 1.0997464848810883e+00, 1.1009410000888527e+00,
      1.1043596489127128e+00, 1.1054473585163858e+00, 1.1065106054734706e+00, 1.1075502058565354e+00,
      1.1085669398385030e+00, 1.1095615536452414e+00, 1.1105347613820860e+00, 1.1114872467436825e+00,
      1.1142267823062899e+00, 1.1151026609077994e+00, 1.1159608321503782e+00, 1.1168018276364671e+00,
      1.1176261578999684e+00, 1.1184343134397452e+00, 1.1192267656928732e+00, 1.1200039679517073e+00};
  return result;
}


template <template <typename, typename> typename Container, template <typename> typename Allocator>
class Correlation_Factor_Data_Fixture {
  typedef Dummy_Electron_List<Container, Allocator> Electron_List_Type;
  typedef Dummy_Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
  typedef Correlation_Factor_Data<Container, Allocator> Correlation_Factor_Data_Type;

 public:
  Correlation_Factor_Data_Fixture() :
      electrons(8),
      electron_pairs(10),
      correlation_factor(CORRELATION_FACTORS::Rational),
      gamma(-1.0),
      beta(-1.0),
      electron_list(new Electron_List_Type(electrons)),
      electron_pair_list(new Electron_Pair_List_Type(electron_pairs)),
      correlation_factor_data(electrons, electron_pairs, correlation_factor, gamma, beta) {
    correlation_factor_data.update(electron_pair_list.get(), electron_list.get());
  }

  int electrons;
  int electron_pairs;
  CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor;
  double gamma;
  double beta;

  std::shared_ptr<Electron_List_Type> electron_list;
  std::shared_ptr<Electron_Pair_List_Type> electron_pair_list;
  Correlation_Factor_Data_Type correlation_factor_data;
};

template <typename T>
class CorrelationFactorDataTest : public testing::Test {
 public:
  T correlation_factor_data_fixture;

  void check(const std::vector<double>& reference, const std::vector<double> trial, std::string test_name) {
    ASSERT_EQ(reference.size(), trial.size());
    for (int i = 0; i < reference.size(); ++i) {
      ASSERT_FLOAT_EQ(trial[i], reference[i]) << test_name << "[" << i << "]";
    }
  }
};

using Implementations = testing::Types<
    Correlation_Factor_Data_Fixture<std::vector, std::allocator>,
#ifdef HAVE_CUDA
    Correlation_Factor_Data_Fixture<thrust::device_vector, thrust::device_allocator>
#endif
    >;
TYPED_TEST_SUITE(CorrelationFactorDataTest, Implementations);

TYPED_TEST(CorrelationFactorDataTest, f12pTest) {
  this->check(f12p_result(),
      get_vector(this->correlation_factor_data_fixture.correlation_factor_data.f12p),
      "f12p_Test");
}
TYPED_TEST(CorrelationFactorDataTest, f12paTest) {
  this->check(f12pa_result(),
      get_vector(this->correlation_factor_data_fixture.correlation_factor_data.f12p_a),
      "f12pa_Test");
}
TYPED_TEST(CorrelationFactorDataTest, f12pcTest) {
  this->check(f12pc_result(),
      get_vector(this->correlation_factor_data_fixture.correlation_factor_data.f12p_c),
      "f12pc_Test");
}
TYPED_TEST(CorrelationFactorDataTest, f12oTest) {
  this->check(f12o_result(),
      get_vector(this->correlation_factor_data_fixture.correlation_factor_data.f12o),
      "f12o_Test");
}
TYPED_TEST(CorrelationFactorDataTest, f12obTest) {
  this->check(f12ob_result(),
      get_vector(this->correlation_factor_data_fixture.correlation_factor_data.f12o_b),
      "f12ob_Test");
}
TYPED_TEST(CorrelationFactorDataTest, f12odTest) {
  this->check(f12od_result(),
      get_vector(this->correlation_factor_data_fixture.correlation_factor_data.f12o_d),
      "f12od_Test");
}
TYPED_TEST(CorrelationFactorDataTest, f13Test) {
  this->check(f13_result(),
      get_vector(this->correlation_factor_data_fixture.correlation_factor_data.f13),
      "f13_Test");
}
TYPED_TEST(CorrelationFactorDataTest, f23Test) {
  this->check(f23_result(),
      get_vector(this->correlation_factor_data_fixture.correlation_factor_data.f23),
      "f23_Test");
}
}  // namespace

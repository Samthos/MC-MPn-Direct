#include <thrust/device_vector.h>
#include "gtest/gtest.h"

#include "../../src/basis/dummy_basis_parser.h"
#include "../../src/basis/basis.h"

#define NWALKERS 5 

namespace {
  std::vector<std::array<double, 3>> create_pos() {
    std::vector<std::array<double, 3>> pos(NWALKERS);
    for (int i = 0; i < NWALKERS; i++) {
      pos[i][0] = i - 2;
      pos[i][1] = i - 2;
      pos[i][2] = i - 2;
    }
    return pos;
  }

  std::vector<double> known_contraction_amplitudes() {
    std::vector<double> known_amp = {
      2.0400914811240242e-14, -3.2300132846920317e-14, 8.3131191032716759e-05, 8.6361291985046149e-03, 3.6157153515072142e-02,
      1.6035067404299949e-12,  6.3630821199921214e-05, 9.4173590343003353e-03, 2.4150463718367900e-02, 1.1277776083265734e-09,
      3.2029654243824013e-04,  8.0272305395771171e-03, 1.3024120378185848e-07, 8.5758954309730960e-04, 6.1750495777277029e-06,
      1.5797071978491031e-04, -2.5010177303378338e-04, 6.2312665009171195e-02, 1.1797284525171936e-01, 9.8275505326898105e-02,
      2.8890484004570741e-03,  9.5980273313993197e-02, 1.2691971129196050e-01, 5.9775960481895568e-02, 1.8854251268197671e-02,
      1.1022150058488693e-01,  6.2479567247585729e-02, 4.5827190181148551e-02, 6.7448902960946050e-02, 5.5244104691114466e-02,
      7.7818733787091530e+00, -5.8858974100643078e+00, 5.6603997097652192e-01, 2.8201336769754959e-01, 1.3714994457481877e-01,
      1.1322210527390615e+01,  1.1007452000846352e+00, 3.0203856019464992e-01, 8.0858731906040621e-02, 4.8210085116246155e+00,
      7.7239183752070772e-01,  1.2382132254845879e-01, 3.2353020987576109e+00, 2.8898370950917501e-01, 1.1468354346860725e+00,
      1.5797071978491031e-04, -2.5010177303378338e-04, 6.2312665009171195e-02, 1.1797284525171936e-01, 9.8275505326898105e-02,
      2.8890484004570741e-03,  9.5980273313993197e-02, 1.2691971129196050e-01, 5.9775960481895568e-02, 1.8854251268197671e-02,
      1.1022150058488693e-01,  6.2479567247585729e-02, 4.5827190181148551e-02, 6.7448902960946050e-02, 5.5244104691114466e-02,
      2.0400914811240242e-14, -3.2300132846920317e-14, 8.3131191032716759e-05, 8.6361291985046149e-03, 3.6157153515072142e-02,
      1.6035067404299949e-12,  6.3630821199921214e-05, 9.4173590343003353e-03, 2.4150463718367900e-02, 1.1277776083265734e-09,
      3.2029654243824013e-04,  8.0272305395771171e-03, 1.3024120378185848e-07, 8.5758954309730960e-04, 6.1750495777277029e-06};
    return known_amp;
  }

  std::vector<double> known_contraction_amplitudes_derivative() {
    std::vector<double> known_amp = {
      -1.0322862894487562e-13, +1.6343867220541680e-13, -1.2228598200912636e-04, -5.0175910643311811e-03, -8.0341195110490305e-03, 
      -7.5942079226764559e-12, -1.0348916759955187e-04, -5.4432335218255936e-03, -4.8639033928792952e-03, -4.1682660403750158e-09, 
      -4.1574491208483573e-04, -3.6604171260471653e-03, -3.6962453633291438e-07, -8.3186185680439028e-04, -1.2485950246165413e-05, 
      -7.9943791406366488e-04, +1.2656244054294412e-03, -9.1661930228490826e-02, -6.8542223091248938e-02, -2.1836817283636761e-02, 
      -1.3682535401940915e-02, -1.5610231651787854e-01, -7.3359593126753164e-02, -1.2038878441053768e-02, -6.9685312687258597e-02, 
      -1.4306750775918325e-01, -2.8490682664899094e-02, -1.3005756573409960e-01, -6.5425435872117660e-02, -1.1170357968543344e-01, 
      -1.6429816901662598e+04, +1.0094338073230063e+04, -8.3264479730646379e-01, -1.6384976663227629e-01, -3.0474717684524734e-02, 
      -3.4118285108684086e+02, -1.7902519934176506e+00, -1.7457828779250764e-01, -1.6284948605876582e-02, -1.7818447458964581e+01, 
      -1.0025646051018786e+00, -5.6462523082097212e-02, -9.1817873562740999e+00, -2.8031419822389975e-01, -2.3189012489352385e+00, 
      -7.9943791406366488e-04, +1.2656244054294412e-03, -9.1661930228490826e-02, -6.8542223091248938e-02, -2.1836817283636761e-02, 
      -1.3682535401940915e-02, -1.5610231651787854e-01, -7.3359593126753164e-02, -1.2038878441053768e-02, -6.9685312687258597e-02, 
      -1.4306750775918325e-01, -2.8490682664899094e-02, -1.3005756573409960e-01, -6.5425435872117660e-02, -1.1170357968543344e-01, 
      -1.0322862894487562e-13, +1.6343867220541680e-13, -1.2228598200912636e-04, -5.0175910643311811e-03, -8.0341195110490305e-03, 
      -7.5942079226764559e-12, -1.0348916759955187e-04, -5.4432335218255936e-03, -4.8639033928792952e-03, -4.1682660403750158e-09, 
      -4.1574491208483573e-04, -3.6604171260471653e-03, -3.6962453633291438e-07, -8.3186185680439028e-04, -1.2485950246165413e-05};
    return known_amp;
  }

  std::vector<double> known_ao_amplitudes() {
    std::vector<double> known_amp = {
      +2.040091481124e-14, -3.230013284692e-14, +8.313119103272e-05, +8.636129198505e-03, +3.615715351507e-02, 
      -3.207013480860e-12, -3.207013480860e-12, -3.207013480860e-12, -1.272616423998e-04, -1.272616423998e-04, 
      -1.272616423998e-04, -1.883471806860e-02, -1.883471806860e-02, -1.883471806860e-02, -4.830092743674e-02, 
      -4.830092743674e-02, -4.830092743674e-02, +7.813472469041e-09, +7.813472469041e-09, +0.000000000000e+00, 
      -7.813472469041e-09, +0.000000000000e+00, +2.219079539967e-03, +2.219079539967e-03, +0.000000000000e+00, 
      -2.219079539967e-03, +0.000000000000e+00, +5.561428455446e-02, +5.561428455446e-02, +0.000000000000e+00, 
      -5.561428455446e-02, +0.000000000000e+00, -1.647435396611e-06, -4.035376105897e-06, -1.276097971006e-06, 
      +2.083859260510e-06, +1.276097971006e-06, +0.000000000000e+00, -1.647435396611e-06, -1.084774501492e-02, 
      -2.657144014638e-02, -8.402627157340e-03, +1.372143268956e-02, +8.402627157340e-03, +0.000000000000e+00, 
      -1.084774501492e-02, +0.000000000000e+00, +4.133133714841e-04, +4.418505798474e-04, -1.562177706406e-04, 
      -3.458027763528e-04, +1.562177706406e-04, +0.000000000000e+00, +4.133133714841e-04, -2.922566877315e-04, 
      +1.579707197849e-04, -2.501017730338e-04, +6.231266500917e-02, +1.179728452517e-01, +9.827550532690e-02, 
      -2.889048400457e-03, -2.889048400457e-03, -2.889048400457e-03, -9.598027331399e-02, -9.598027331399e-02, 
      -9.598027331399e-02, -1.269197112920e-01, -1.269197112920e-01, -1.269197112920e-01, -5.977596048190e-02, 
      -5.977596048190e-02, -5.977596048190e-02, +3.265652113519e-02, +3.265652113519e-02, +0.000000000000e+00, 
      -3.265652113519e-02, +0.000000000000e+00, +1.909092390995e-01, +1.909092390995e-01, +0.000000000000e+00, 
      -1.909092390995e-01, +0.000000000000e+00, +1.082177849077e-01, +1.082177849077e-01, +0.000000000000e+00, 
      -1.082177849077e-01, +0.000000000000e+00, -7.245914986907e-02, -1.774879443751e-01, -5.612661614465e-02, 
      +9.165438036230e-02, +5.612661614465e-02, +0.000000000000e+00, -7.245914986907e-02, -1.066460795181e-01, 
      -2.612284778877e-01, -8.260769798241e-02, +1.348978059219e-01, +8.260769798241e-02, +0.000000000000e+00, 
      -1.066460795181e-01, +0.000000000000e+00, +2.311026704836e-01, +2.470591468909e-01, -8.734859906036e-02, 
      -1.933543664189e-01, +8.734859906036e-02, +0.000000000000e+00, +2.311026704836e-01, -1.634142654493e-01, 
      +7.781873378709e+00, -5.885897410064e+00, +5.660399709765e-01, +2.820133676975e-01, +1.371499445748e-01, 
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, 
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, 
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, 
      -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, 
      -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, 
      -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, 
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, 
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, 
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, 
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, 
      +1.579707197849e-04, -2.501017730338e-04, +6.231266500917e-02, +1.179728452517e-01, +9.827550532690e-02, 
      +2.889048400457e-03, +2.889048400457e-03, +2.889048400457e-03, +9.598027331399e-02, +9.598027331399e-02, 
      +9.598027331399e-02, +1.269197112920e-01, +1.269197112920e-01, +1.269197112920e-01, +5.977596048190e-02, 
      +5.977596048190e-02, +5.977596048190e-02, +3.265652113519e-02, +3.265652113519e-02, +0.000000000000e+00, 
      -3.265652113519e-02, +0.000000000000e+00, +1.909092390995e-01, +1.909092390995e-01, +0.000000000000e+00, 
      -1.909092390995e-01, +0.000000000000e+00, +1.082177849077e-01, +1.082177849077e-01, +0.000000000000e+00, 
      -1.082177849077e-01, +0.000000000000e+00, +7.245914986907e-02, +1.774879443751e-01, +5.612661614465e-02, 
      -9.165438036230e-02, -5.612661614465e-02, +0.000000000000e+00, +7.245914986907e-02, +1.066460795181e-01, 
      +2.612284778877e-01, +8.260769798241e-02, -1.348978059219e-01, -8.260769798241e-02, +0.000000000000e+00, 
      +1.066460795181e-01, +0.000000000000e+00, +2.311026704836e-01, +2.470591468909e-01, -8.734859906036e-02, 
      -1.933543664189e-01, +8.734859906036e-02, +0.000000000000e+00, +2.311026704836e-01, -1.634142654493e-01, 
      +2.040091481124e-14, -3.230013284692e-14, +8.313119103272e-05, +8.636129198505e-03, +3.615715351507e-02, 
      +3.207013480860e-12, +3.207013480860e-12, +3.207013480860e-12, +1.272616423998e-04, +1.272616423998e-04, 
      +1.272616423998e-04, +1.883471806860e-02, +1.883471806860e-02, +1.883471806860e-02, +4.830092743674e-02, 
      +4.830092743674e-02, +4.830092743674e-02, +7.813472469041e-09, +7.813472469041e-09, +0.000000000000e+00, 
      -7.813472469041e-09, +0.000000000000e+00, +2.219079539967e-03, +2.219079539967e-03, +0.000000000000e+00, 
      -2.219079539967e-03, +0.000000000000e+00, +5.561428455446e-02, +5.561428455446e-02, +0.000000000000e+00, 
      -5.561428455446e-02, +0.000000000000e+00, +1.647435396611e-06, +4.035376105897e-06, +1.276097971006e-06, 
      -2.083859260510e-06, -1.276097971006e-06, +0.000000000000e+00, +1.647435396611e-06, +1.084774501492e-02, 
      +2.657144014638e-02, +8.402627157340e-03, -1.372143268956e-02, -8.402627157340e-03, +0.000000000000e+00, 
      +1.084774501492e-02, +0.000000000000e+00, +4.133133714841e-04, +4.418505798474e-04, -1.562177706406e-04, 
      -3.458027763528e-04, +1.562177706406e-04, +0.000000000000e+00, +4.133133714841e-04, -2.922566877315e-04};
    return known_amp;
  }

  std::vector<double> known_ao_amplitudes_dx() {
    std::vector<double> known_amp = {
      +2.064572578898e-13, -3.268773444108e-13, +2.445719640183e-04, +1.003518212866e-02, +1.606823902210e-02,
      -2.877332495028e-11, -3.037683169071e-11, -3.037683169071e-11, -3.503258491983e-04, -4.139566703982e-04,
      -4.139566703982e-04, -1.235557505300e-02, -2.177293408730e-02, -2.177293408730e-02, +4.694850146851e-03,
      -1.945561357152e-02, -1.945561357152e-02, +5.385045225663e-08, +5.775718849115e-08, +2.255555216653e-09,
      -5.385045225663e-08, -3.906736234520e-09, +4.651190715770e-03, +5.760730485754e-03, +6.405930848765e-04,
      -4.651190715770e-03, -1.109539769983e-03, +2.291308523644e-02, +5.072022751367e-02, +1.605446107915e-02,
      -2.291308523644e-02, -2.780714227723e-02, -6.879690216249e-06, -2.088710672412e-05, -7.881181068930e-06,
      +1.026509071727e-05, +7.243132083428e-06, +2.017688052949e-06, -9.350843311165e-06, -4.773007806566e-03,
      -3.826287381079e-02, -2.050241026391e-02, +1.632850490057e-02, +1.630109668524e-02, +1.328572007319e-02,
      -2.104462532895e-02, -2.922566877315e-04, +1.051469217055e-03, +1.676381099941e-03, -3.974180085098e-04,
      -1.250225237705e-03, +3.193091231895e-04, -2.209252899237e-04, +1.671439274282e-03, -8.896293574546e-04,
      +7.994379140637e-04, -1.265624405429e-03, +9.166193022849e-02, +6.854222309125e-02, +2.183681728364e-02,
      -1.079348700148e-02, -1.368253540194e-02, -1.368253540194e-02, -6.012204320389e-02, -1.561023165179e-01,
      -1.561023165179e-01, +5.356011816521e-02, -7.335959312675e-02, -7.335959312675e-02, +4.773708204084e-02,
      -1.203887844105e-02, -1.203887844105e-02, +8.804198098047e-02, +1.206985021157e-01, +1.885425126820e-02,
      -8.804198098047e-02, -3.265652113519e-02, +5.689095325165e-02, +2.478001923512e-01, +1.102215005849e-01,
      -5.689095325165e-02, -1.909092390995e-01, -5.887047498981e-02, +4.934730991793e-02, +6.247956724759e-02,
      +5.887047498981e-02, -1.082177849077e-01, +1.173838227879e-02, -3.262228417614e-01, -2.154139527632e-01,
      +1.226335609248e-01, +1.592873366185e-01, +1.774879443751e-01, -2.056390673284e-01, +2.164915414218e-01,
      +7.836854336631e-03, -1.627371650253e-01, -7.149583713860e-02, +8.012946704294e-02, +2.612284778877e-01,
      -1.034466971326e-01, -3.268285308985e-01, -2.260184117330e-01, +3.760240215679e-01, +8.542692988103e-02,
      -2.252302148257e-01, -1.727755289414e-01, -2.470591468909e-01, +4.672895997178e-01, -3.595113839884e-03,
      -0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
      +1.132221052739e+01, -0.000000000000e+00, -0.000000000000e+00, +1.100745200085e+00, -0.000000000000e+00,
      -0.000000000000e+00, +3.020385601946e-01, -0.000000000000e+00, -0.000000000000e+00, +8.085873190604e-02,
      -0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00,
      -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00,
      -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00,
      -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00,
      -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      -7.994379140637e-04, +1.265624405429e-03, -9.166193022849e-02, -6.854222309125e-02, -2.183681728364e-02,
      -1.079348700148e-02, -1.368253540194e-02, -1.368253540194e-02, -6.012204320389e-02, -1.561023165179e-01,
      -1.561023165179e-01, +5.356011816521e-02, -7.335959312675e-02, -7.335959312675e-02, +4.773708204084e-02,
      -1.203887844105e-02, -1.203887844105e-02, -8.804198098047e-02, -1.206985021157e-01, -1.885425126820e-02,
      +8.804198098047e-02, +3.265652113519e-02, -5.689095325165e-02, -2.478001923512e-01, -1.102215005849e-01,
      +5.689095325165e-02, +1.909092390995e-01, +5.887047498981e-02, -4.934730991793e-02, -6.247956724759e-02,
      -5.887047498981e-02, +1.082177849077e-01, +1.173838227879e-02, -3.262228417614e-01, -2.154139527632e-01,
      +1.226335609248e-01, +1.592873366185e-01, +1.774879443751e-01, -2.056390673284e-01, +2.164915414218e-01,
      +7.836854336631e-03, -1.627371650253e-01, -7.149583713860e-02, +8.012946704294e-02, +2.612284778877e-01,
      -1.034466971326e-01, +3.268285308985e-01, +2.260184117330e-01, -3.760240215679e-01, -8.542692988103e-02,
      +2.252302148257e-01, +1.727755289414e-01, +2.470591468909e-01, -4.672895997178e-01, +3.595113839884e-03,
      -2.064572578898e-13, +3.268773444108e-13, -2.445719640183e-04, -1.003518212866e-02, -1.606823902210e-02,
      -2.877332495028e-11, -3.037683169071e-11, -3.037683169071e-11, -3.503258491983e-04, -4.139566703982e-04,
      -4.139566703982e-04, -1.235557505300e-02, -2.177293408730e-02, -2.177293408730e-02, +4.694850146851e-03,
      -1.945561357152e-02, -1.945561357152e-02, -5.385045225663e-08, -5.775718849115e-08, -2.255555216653e-09,
      +5.385045225663e-08, +3.906736234520e-09, -4.651190715770e-03, -5.760730485754e-03, -6.405930848765e-04,
      +4.651190715770e-03, +1.109539769983e-03, -2.291308523644e-02, -5.072022751367e-02, -1.605446107915e-02,
      +2.291308523644e-02, +2.780714227723e-02, -6.879690216249e-06, -2.088710672412e-05, -7.881181068930e-06,
      +1.026509071727e-05, +7.243132083428e-06, +2.017688052949e-06, -9.350843311165e-06, -4.773007806566e-03,
      -3.826287381079e-02, -2.050241026391e-02, +1.632850490057e-02, +1.630109668524e-02, +1.328572007319e-02,
      -2.104462532895e-02, +2.922566877315e-04, -1.051469217055e-03, -1.676381099941e-03, +3.974180085098e-04,
      +1.250225237705e-03, -3.193091231895e-04, +2.209252899237e-04, -1.671439274282e-03, +8.896293574546e-04};
    return known_amp;
  }

  std::vector<double> known_ao_amplitudes_dy() {
    std::vector<double> known_amp = {
      +2.064572578898e-13, -3.268773444108e-13, +2.445719640183e-04, +1.003518212866e-02, +1.606823902210e-02,
      -3.037683169071e-11, -2.877332495028e-11, -3.037683169071e-11, -4.139566703982e-04, -3.503258491983e-04,
      -4.139566703982e-04, -2.177293408730e-02, -1.235557505300e-02, -2.177293408730e-02, -1.945561357152e-02,
      +4.694850146851e-03, -1.945561357152e-02, +5.385045225663e-08, +5.385045225663e-08, +2.255555216653e-09,
      -5.775718849115e-08, +3.906736234520e-09, +4.651190715770e-03, +4.651190715770e-03, +6.405930848765e-04,
      -5.760730485754e-03, +1.109539769983e-03, +2.291308523644e-02, +2.291308523644e-02, +1.605446107915e-02,
      -5.072022751367e-02, +2.780714227723e-02, -9.350843311165e-06, -2.088710672412e-05, -7.243132083428e-06,
      +1.026509071727e-05, +7.881181068930e-06, -2.017688052949e-06, -6.879690216249e-06, -2.104462532895e-02,
      -3.826287381079e-02, -1.630109668524e-02, +1.632850490057e-02, +2.050241026391e-02, -1.328572007319e-02,
      -4.773007806566e-03, +2.922566877315e-04, +1.671439274282e-03, +1.676381099941e-03, -3.193091231895e-04,
      -1.250225237705e-03, +3.974180085098e-04, +2.209252899237e-04, +1.051469217055e-03, -8.896293574546e-04,
      +7.994379140637e-04, -1.265624405429e-03, +9.166193022849e-02, +6.854222309125e-02, +2.183681728364e-02,
      -1.368253540194e-02, -1.079348700148e-02, -1.368253540194e-02, -1.561023165179e-01, -6.012204320389e-02,
      -1.561023165179e-01, -7.335959312675e-02, +5.356011816521e-02, -7.335959312675e-02, -1.203887844105e-02,
      +4.773708204084e-02, -1.203887844105e-02, +8.804198098047e-02, +8.804198098047e-02, +1.885425126820e-02,
      -1.206985021157e-01, +3.265652113519e-02, +5.689095325165e-02, +5.689095325165e-02, +1.102215005849e-01,
      -2.478001923512e-01, +1.909092390995e-01, -5.887047498981e-02, -5.887047498981e-02, +6.247956724759e-02,
      -4.934730991793e-02, +1.082177849077e-01, -2.056390673284e-01, -3.262228417614e-01, -1.592873366185e-01,
      +1.226335609248e-01, +2.154139527632e-01, -1.774879443751e-01, +1.173838227879e-02, -1.034466971326e-01,
      +7.836854336631e-03, -8.012946704294e-02, -7.149583713860e-02, +1.627371650253e-01, -2.612284778877e-01,
      +2.164915414218e-01, +3.268285308985e-01, +4.672895997178e-01, +3.760240215679e-01, +1.727755289414e-01,
      -2.252302148257e-01, -8.542692988103e-02, +2.470591468909e-01, -2.260184117330e-01, -3.595113839884e-03,
      -0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
      -0.000000000000e+00, +1.132221052739e+01, -0.000000000000e+00, -0.000000000000e+00, +1.100745200085e+00,
      -0.000000000000e+00, -0.000000000000e+00, +3.020385601946e-01, -0.000000000000e+00, -0.000000000000e+00,
      +8.085873190604e-02, -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      -0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00,
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      -7.994379140637e-04, +1.265624405429e-03, -9.166193022849e-02, -6.854222309125e-02, -2.183681728364e-02,
      -1.368253540194e-02, -1.079348700148e-02, -1.368253540194e-02, -1.561023165179e-01, -6.012204320389e-02,
      -1.561023165179e-01, -7.335959312675e-02, +5.356011816521e-02, -7.335959312675e-02, -1.203887844105e-02,
      +4.773708204084e-02, -1.203887844105e-02, -8.804198098047e-02, -8.804198098047e-02, -1.885425126820e-02,
      +1.206985021157e-01, -3.265652113519e-02, -5.689095325165e-02, -5.689095325165e-02, -1.102215005849e-01,
      +2.478001923512e-01, -1.909092390995e-01, +5.887047498981e-02, +5.887047498981e-02, -6.247956724759e-02,
      +4.934730991793e-02, -1.082177849077e-01, -2.056390673284e-01, -3.262228417614e-01, -1.592873366185e-01,
      +1.226335609248e-01, +2.154139527632e-01, -1.774879443751e-01, +1.173838227879e-02, -1.034466971326e-01,
      +7.836854336631e-03, -8.012946704294e-02, -7.149583713860e-02, +1.627371650253e-01, -2.612284778877e-01,
      +2.164915414218e-01, -3.268285308985e-01, -4.672895997178e-01, -3.760240215679e-01, -1.727755289414e-01,
      +2.252302148257e-01, +8.542692988103e-02, -2.470591468909e-01, +2.260184117330e-01, +3.595113839884e-03,
      -2.064572578898e-13, +3.268773444108e-13, -2.445719640183e-04, -1.003518212866e-02, -1.606823902210e-02,
      -3.037683169071e-11, -2.877332495028e-11, -3.037683169071e-11, -4.139566703982e-04, -3.503258491983e-04,
      -4.139566703982e-04, -2.177293408730e-02, -1.235557505300e-02, -2.177293408730e-02, -1.945561357152e-02,
      +4.694850146851e-03, -1.945561357152e-02, -5.385045225663e-08, -5.385045225663e-08, -2.255555216653e-09,
      +5.775718849115e-08, -3.906736234520e-09, -4.651190715770e-03, -4.651190715770e-03, -6.405930848765e-04,
      +5.760730485754e-03, -1.109539769983e-03, -2.291308523644e-02, -2.291308523644e-02, -1.605446107915e-02,
      +5.072022751367e-02, -2.780714227723e-02, -9.350843311165e-06, -2.088710672412e-05, -7.243132083428e-06,
      +1.026509071727e-05, +7.881181068930e-06, -2.017688052949e-06, -6.879690216249e-06, -2.104462532895e-02,
      -3.826287381079e-02, -1.630109668524e-02, +1.632850490057e-02, +2.050241026391e-02, -1.328572007319e-02,
      -4.773007806566e-03, -2.922566877315e-04, -1.671439274282e-03, -1.676381099941e-03, +3.193091231895e-04,
      +1.250225237705e-03, -3.974180085098e-04, -2.209252899237e-04, -1.051469217055e-03, +8.896293574546e-04};
    return known_amp;
  }

  std::vector<double> known_ao_amplitudes_dz() {
    std::vector<double> known_amp = {
      +2.064572578898e-13, -3.268773444108e-13, +2.445719640183e-04, +1.003518212866e-02, +1.606823902210e-02,
      -3.037683169071e-11, -3.037683169071e-11, -2.877332495028e-11, -4.139566703982e-04, -4.139566703982e-04,
      -3.503258491983e-04, -2.177293408730e-02, -2.177293408730e-02, -1.235557505300e-02, -1.945561357152e-02,
      -1.945561357152e-02, +4.694850146851e-03, +5.775718849115e-08, +5.385045225663e-08, -4.511110433306e-09,
      -5.385045225663e-08, +0.000000000000e+00, +5.760730485754e-03, +4.651190715770e-03, -1.281186169753e-03,
      -4.651190715770e-03, +0.000000000000e+00, +5.072022751367e-02, +2.291308523644e-02, -3.210892215831e-02,
      -2.291308523644e-02, +0.000000000000e+00, -9.350843311165e-06, -2.088710672412e-05, -4.690936141417e-06,
      +1.182798516265e-05, +4.690936141417e-06, +0.000000000000e+00, -9.350843311165e-06, -2.104462532895e-02,
      -3.826287381079e-02, +5.041576294404e-04, +2.661957941774e-02, -5.041576294404e-04, +0.000000000000e+00,
      -2.104462532895e-02, +0.000000000000e+00, +1.464782588540e-03, +1.124067875132e-03, -8.660713204317e-04,
      -1.003223254596e-03, +8.660713204317e-04, +0.000000000000e+00, +1.464782588540e-03, -1.181886045186e-03,
      +7.994379140637e-04, -1.265624405429e-03, +9.166193022849e-02, +6.854222309125e-02, +2.183681728364e-02,
      -1.368253540194e-02, -1.368253540194e-02, -1.079348700148e-02, -1.561023165179e-01, -1.561023165179e-01,
      -6.012204320389e-02, -7.335959312675e-02, -7.335959312675e-02, +5.356011816521e-02, -1.203887844105e-02,
      -1.203887844105e-02, +4.773708204084e-02, +1.206985021157e-01, +8.804198098047e-02, -3.770850253640e-02,
      -8.804198098047e-02, +0.000000000000e+00, +2.478001923512e-01, +5.689095325165e-02, -2.204430011698e-01,
      -5.689095325165e-02, +0.000000000000e+00, +4.934730991793e-02, -5.887047498981e-02, -1.249591344952e-01,
      +5.887047498981e-02, +0.000000000000e+00, -2.056390673284e-01, -3.262228417614e-01, +6.521912796008e-02,
      +2.601151314682e-01, -6.521912796008e-02, +0.000000000000e+00, -2.056390673284e-01, -1.034466971326e-01,
      +7.836854336631e-03, +2.503013248867e-01, +1.308508717442e-01, -2.503013248867e-01, +0.000000000000e+00,
      -1.034466971326e-01, +0.000000000000e+00, +2.361869292342e-01, -2.416238456593e-01, -4.386646644811e-01,
      +5.099030862990e-02, +4.386646644811e-01, +0.000000000000e+00, +2.361869292342e-01, -3.304236447384e-01,
      -0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
      -0.000000000000e+00, -0.000000000000e+00, +1.132221052739e+01, -0.000000000000e+00, -0.000000000000e+00,
      +1.100745200085e+00, -0.000000000000e+00, -0.000000000000e+00, +3.020385601946e-01, -0.000000000000e+00,
      -0.000000000000e+00, +8.085873190604e-02, -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      -0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      -0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, -0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00,
      -7.994379140637e-04, +1.265624405429e-03, -9.166193022849e-02, -6.854222309125e-02, -2.183681728364e-02,
      -1.368253540194e-02, -1.368253540194e-02, -1.079348700148e-02, -1.561023165179e-01, -1.561023165179e-01,
      -6.012204320389e-02, -7.335959312675e-02, -7.335959312675e-02, +5.356011816521e-02, -1.203887844105e-02,
      -1.203887844105e-02, +4.773708204084e-02, -1.206985021157e-01, -8.804198098047e-02, +3.770850253640e-02,
      +8.804198098047e-02, +0.000000000000e+00, -2.478001923512e-01, -5.689095325165e-02, +2.204430011698e-01,
      +5.689095325165e-02, +0.000000000000e+00, -4.934730991793e-02, +5.887047498981e-02, +1.249591344952e-01,
      -5.887047498981e-02, +0.000000000000e+00, -2.056390673284e-01, -3.262228417614e-01, +6.521912796008e-02,
      +2.601151314682e-01, -6.521912796008e-02, +0.000000000000e+00, -2.056390673284e-01, -1.034466971326e-01,
      +7.836854336631e-03, +2.503013248867e-01, +1.308508717442e-01, -2.503013248867e-01, +0.000000000000e+00,
      -1.034466971326e-01, +0.000000000000e+00, -2.361869292342e-01, +2.416238456593e-01, +4.386646644811e-01,
      -5.099030862990e-02, -4.386646644811e-01, +0.000000000000e+00, -2.361869292342e-01, +3.304236447384e-01,
      -2.064572578898e-13, +3.268773444108e-13, -2.445719640183e-04, -1.003518212866e-02, -1.606823902210e-02,
      -3.037683169071e-11, -3.037683169071e-11, -2.877332495028e-11, -4.139566703982e-04, -4.139566703982e-04,
      -3.503258491983e-04, -2.177293408730e-02, -2.177293408730e-02, -1.235557505300e-02, -1.945561357152e-02,
      -1.945561357152e-02, +4.694850146851e-03, -5.775718849115e-08, -5.385045225663e-08, +4.511110433306e-09,
      +5.385045225663e-08, +0.000000000000e+00, -5.760730485754e-03, -4.651190715770e-03, +1.281186169753e-03,
      +4.651190715770e-03, +0.000000000000e+00, -5.072022751367e-02, -2.291308523644e-02, +3.210892215831e-02,
      +2.291308523644e-02, +0.000000000000e+00, -9.350843311165e-06, -2.088710672412e-05, -4.690936141417e-06,
      +1.182798516265e-05, +4.690936141417e-06, +0.000000000000e+00, -9.350843311165e-06, -2.104462532895e-02,
      -3.826287381079e-02, +5.041576294404e-04, +2.661957941774e-02, -5.041576294404e-04, +0.000000000000e+00,
      -2.104462532895e-02, +0.000000000000e+00, -1.464782588540e-03, -1.124067875132e-03, +8.660713204317e-04,
      +1.003223254596e-03, -8.660713204317e-04, +0.000000000000e+00, -1.464782588540e-03, +1.181886045186e-03};
    return known_amp;
  }

  void compare_vector_double(const std::vector<double>& trial, const std::vector<double>& known, const std::string& array_name) {
    ASSERT_EQ(trial.size(), known.size());
    for (int i = 0; i < known.size(); i++) {
      ASSERT_FLOAT_EQ(trial[i], known[i]) << array_name << i/5 << " " << i%5;
    }
  }

  void print_vector_double(const std::vector<double>& trial, bool transpose = true) {
    auto other = trial.size() / NWALKERS;
    if (transpose) {
      for (int j = 0; j < other; j++) {
        for (int i = 0; i < 5; i++) {
          printf("%+16.12e, ", trial[i * other + j]);
        }
        printf("\n");
      }
    } else {
      for (int i = 0, idx = 1; i < 5; i++) {
        for (int j = 0; j < other; j++, idx++) {
          printf("%+16.12e, ", trial[i * other + j]);
          if (idx % 5 == 0) {
            printf("\n");
          }
        }
      }
    }
  }

  template <class T>
  class BasisTest : public testing::Test {
    public:
    BasisTest() : basis(NWALKERS, Dummy_Basis_Parser(true)), pos(create_pos()) { }
   
    T basis;
    std::vector<std::array<double, 3>> pos;
  };


#define MYVAR
#ifdef MYVAR
   using Implementations = testing::Types<Basis_Host, Basis_Device>;
#else
  using Implementations = testing::Types<Basis_Host>;
#endif
  TYPED_TEST_SUITE(BasisTest, Implementations);

  TYPED_TEST(BasisTest, BuildContractionTest) {
    this->basis.build_contractions(this->pos);
    compare_vector_double(this->basis.get_contraction_amplitudes(), known_contraction_amplitudes(), "contraction_amplitudes");
  }

  TYPED_TEST(BasisTest, BuildContractionWithDerivativeTest) {
    this->basis.build_contractions_with_derivatives(this->pos);
    compare_vector_double(this->basis.get_contraction_amplitudes(), known_contraction_amplitudes(), "contraction_amplitudes");
    compare_vector_double(this->basis.get_contraction_amplitudes_derivative(), known_contraction_amplitudes_derivative(), "contraction_amplitudes_derivative");
  }

  TYPED_TEST(BasisTest, BuildAOTest) {
    this->basis.build_contractions(this->pos);
    this->basis.build_ao_amplitudes(this->pos);
    compare_vector_double(this->basis.get_ao_amplitudes(), known_ao_amplitudes(), "ao_amplitudes");
  }

  TYPED_TEST(BasisTest, BuildAODXTest) {
    this->basis.build_contractions_with_derivatives(this->pos);
    this->basis.build_ao_amplitudes_dx(this->pos);
    compare_vector_double(this->basis.get_ao_amplitudes(), known_ao_amplitudes_dx(), "ao_amplitudes_dx");
  }

  TYPED_TEST(BasisTest, BuildAODYTest) {
    this->basis.build_contractions_with_derivatives(this->pos);
    this->basis.build_ao_amplitudes_dy(this->pos);
    compare_vector_double(this->basis.get_ao_amplitudes(), known_ao_amplitudes_dy(), "ao_amplitudes_dy");
  }

  TYPED_TEST(BasisTest, BuildAODZTest) {
    this->basis.build_contractions_with_derivatives(this->pos);
    this->basis.build_ao_amplitudes_dz(this->pos);
    compare_vector_double(this->basis.get_ao_amplitudes(), known_ao_amplitudes_dz(), "ao_amplitudes_dz");
  }
}

//
// Created by aedoran on 1/2/20.
//

#ifndef F12_METHODS_SRC_F12_TRACES_H_
#define F12_METHODS_SRC_F12_TRACES_H_

#include <vector>
#include "../basis/qc_basis.h"

class F12_Traces {
 public:
  F12_Traces(int io1, int io2, int iv1, int iv2, int electron_pairs_, int electrons_) :
    iocc1(io1),
    iocc2(io2),
    ivir1(iv1),
    ivir2(iv2),
    electron_pairs(electron_pairs_),
    electrons(electrons_),
    op11(electrons, 0.0),
    op12(electrons, std::vector<double>(electrons, 0.0)),
    ok12(electrons, std::vector<double>(electrons, 0.0)),
    ov12(electrons, std::vector<double>(electrons, 0.0)),
    dop11(electrons, std::vector<double>(electrons, 0.0)),
    dop12(electrons, std::vector<double>(electrons, 0.0)),
    p11(electron_pairs, 0.0),
    p12(electron_pairs, 0.0),
    p22(electron_pairs, 0.0),
    k12(electron_pairs, 0.0),
    dp11(electron_pairs, 0.0),
    dp12(electron_pairs, 0.0),
    dp21(electron_pairs, 0.0),
    dp22(electron_pairs, 0.0),
    p13(electron_pairs, std::vector<double>(electrons, 0.0)),
    k13(electron_pairs, std::vector<double>(electrons, 0.0)),
    v13(electron_pairs, std::vector<double>(electrons, 0.0)),
    dp31(electron_pairs, std::vector<double>(electrons, 0.0)),
    p23(electron_pairs, std::vector<double>(electrons, 0.0)),
    k23(electron_pairs, std::vector<double>(electrons, 0.0)),
    v23(electron_pairs, std::vector<double>(electrons, 0.0)),
    dp32(electron_pairs, std::vector<double>(electrons, 0.0)),
    ds_p11(electrons, std::vector<double>(electrons, 0.0)),
    ds_p12(electrons, std::vector<double>(electrons, 0.0)),
    ds_p21(electrons, std::vector<double>(electrons, 0.0)),
    ds_p22(electrons, std::vector<double>(electrons, 0.0)),
    ds_p31(electrons, std::vector<std::vector<double>>(electrons, std::vector<double>(electrons, 0.0))),
    ds_p32(electrons, std::vector<std::vector<double>>(electrons, std::vector<double>(electrons, 0.0)))
  {
  }


  // wavefunction shim
  int iocc1, iocc2, ivir1, ivir2, electron_pairs, electrons;

  // traces of single electrons with themselves
  std::vector<double> op11;

  // traces of single electrons with single electrons
  std::vector<std::vector<double>> op12;
  std::vector<std::vector<double>> ok12;
  std::vector<std::vector<double>> ov12;
  std::vector<std::vector<double>> dop11;
  std::vector<std::vector<double>> dop12;

  // traces of electrons pairs with themselves
  std::vector<double> p11;
  std::vector<double> p12;
  std::vector<double> p22;
  std::vector<double> k12;
  std::vector<double> dp11;
  std::vector<double> dp12;
  std::vector<double> dp21;
  std::vector<double> dp22;

  // traces of electron pairs with single electrons
  std::vector<std::vector<double>> p13;
  std::vector<std::vector<double>> k13;
  std::vector<std::vector<double>> v13;
  std::vector<std::vector<double>> dp31;
  std::vector<std::vector<double>> p23;
  std::vector<std::vector<double>> k23;
  std::vector<std::vector<double>> v23;
  std::vector<std::vector<double>> dp32;

  // extra one electron traces
  std::vector<std::vector<double>> ds_p11;
  std::vector<std::vector<double>> ds_p12;
  std::vector<std::vector<double>> ds_p21;
  std::vector<std::vector<double>> ds_p22;
  std::vector<std::vector<std::vector<double>>> ds_p31;
  std::vector<std::vector<std::vector<double>>> ds_p32;

  void update_v(const Wavefunction& electron_pair_psi1, const Wavefunction& electron_pair_psi2, const Wavefunction& electron_psi);
  // void update_bx(const std::vector<electron_pair_typ>& electron_pair_list, const std::vector<el_one_typ>& el_one_list);
  // void update_bx_fd_traces(const std::vector<el_one_typ>& el_one_list);

 private:
  void build_one_e_traces(const Wavefunction& electron_psi);
  void build_one_e_one_e_traces(const Wavefunction& electron_psi);
  void build_two_e_traces(const Wavefunction& electron_pair_psi1, const Wavefunction& electron_pair_psi2);
  void build_two_e_one_e_traces(const Wavefunction& electron_pair_psi1, const Wavefunction& electron_pair_psi2, const Wavefunction& electron_psi);
  // void build_two_e_derivative_traces(const std::vector<electron_pair_typ>& electron_pair_list);
  // void build_two_e_one_e_derivative_traces(const std::vector<electron_pair_typ>& electron_pair_list, const std::vector<el_one_typ>& el_one_list);
};

#endif //F12_METHODS_SRC_F12_TRACES_H_

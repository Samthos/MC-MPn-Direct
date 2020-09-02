#ifndef X_TRACES_H_
#define X_TRACES_H_
#include <unordered_map>

#include "../basis/wavefunction.h"
#include "electron_pair_list.h"
#include "electron_list.h"

class X_Traces {
    typedef Wavefunction_Host Wavefunction_Type;

  public:
    X_Traces(int n_e_p, int n_e);

    void set(int band, int offBand, std::unordered_map<int, Wavefunction_Type>& wavefunctions);
    void set_derivative_traces(int band, int offBand, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);
    void set_fd_derivative_traces(int band, int offBand, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);

    int n_electron_pairs;
    int n_electrons;
    std::vector<double> x11;
    std::vector<double> x12;
    std::vector<double> x22;

    std::vector<double> dx11;
    std::vector<double> dx12;
    std::vector<double> dx21;
    std::vector<double> dx22;

    std::vector<double> x13;
    std::vector<double> x23;

    std::vector<double> dx31;
    std::vector<double> dx32;

    std::vector<double> ox11;
    std::vector<double> ox12;

    std::vector<std::vector<double>> ds_x11;
    std::vector<std::vector<double>> ds_x22;
    std::vector<std::vector<double>> ds_x12;
    std::vector<std::vector<double>> ds_x21;
    std::vector<std::vector<std::vector<double>>> ds_x31;
    std::vector<std::vector<std::vector<double>>> ds_x32;
  private:
};
#endif  // X_TRACES_H_

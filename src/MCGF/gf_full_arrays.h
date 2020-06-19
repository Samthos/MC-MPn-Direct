#ifndef GF_FULL_ARRAYS_H_
#define GF_FULL_ARRAYS_H_

#include <vector>
#include "../qc_input.h"
#include "../basis/nwchem_movec_parser.h"

class OVPS_ARRAY {
 public:
  OVPS_ARRAY() = default;
  OVPS_ARRAY(const OVPS_ARRAY& other);
  OVPS_ARRAY& operator = (const OVPS_ARRAY& other);
  void resize(const IOPs& iops, const NWChem_Movec_Parser& basis, const std::vector<int>& order);
  void zero_energy_arrays();

  std::vector<std::vector<std::vector<double>>> enBlock;
  std::vector<std::vector<double>> enEx1;
  std::vector<std::vector<std::vector<double>>> enCov;

  std::vector<double> ent;
  std::vector<double> enCore;
  std::vector<double> enGrouped;
          
  std::vector<double> en2mCore;
  std::vector<double> en2pCore;
          
  std::vector<double> en3_1pCore;
  std::vector<double> en3_2pCore;
  std::vector<double> en3_12pCore;
  std::vector<double> en3_1mCore;
  std::vector<double> en3_2mCore;
  std::vector<double> en3_12mCore;
  std::vector<double> en3_12cCore;
  std::vector<double> en3_22cCore;
  std::vector<double> en3c12;
  std::vector<double> en3c22;
  std::vector<double> one;
          
  double* en2m;
  double* en2p;
  double* en3_1p;
  double* en3_2p;
  double* en3_12p;
  double* en3_1m;
  double* en3_2m;
  double* en3_12m;
  double* en3_c;
          
  double* rv;

 private:
  OVPS_ARRAY(int electron_pairs_, int numBand_, int offBand_, int numDiff_, int numBlock_, int nmo_, const std::vector<int>& orders_);
  int electron_pairs;
  int numBand;
  int offBand;
  int numDiff;
  int numBlock;
  int nmo;
};

#endif

#include <iostream>
#include <algorithm>

#include "gf_full_arrays.h"

void OVPS_ARRAY::resize(const IOPs& iops, const std::shared_ptr<Movec_Parser> basis, const std::vector<int>& orders) {
  electron_pairs = iops.iopns[KEYS::ELECTRON_PAIRS];
  numBand = iops.iopns[KEYS::NUM_BAND];
  offBand = iops.iopns[KEYS::OFF_BAND];
  numDiff = iops.iopns[KEYS::DIFFS];
  numBlock = iops.iopns[KEYS::NBLOCK];
  nmo = basis->ivir2 - basis->iocc1;

  enEx1.resize(numBand);
  for (auto &it : enEx1) {
    it.resize(nmo * nmo * numDiff);
    std::fill(it.begin(), it.end(), 0.0);
  }

  enCov.resize(numBand);
  for (auto &it : enCov) {
    it.resize(numBlock);
    for (auto &jt : it) {
      jt.resize(nmo * nmo * numDiff * (numDiff + 1) / 2);
      std::fill(jt.begin(), jt.end(), 0.0);
    }
  }

  enBlock.resize(numBand);
  for (auto &it : enBlock) {
    it.resize(numBlock);
    for (auto &jt : it) {
      jt.resize(nmo * nmo * numDiff);
      std::fill(jt.begin(), jt.end(), 0.0);
    }
  }

  ent.resize(nmo * electron_pairs);
  if (std::find(orders.begin(), orders.end(), 2) != orders.end()) {
    en2mCore.resize(electron_pairs * electron_pairs);
    en2pCore.resize(electron_pairs * electron_pairs);
    enCore.resize(electron_pairs * electron_pairs);
  }
  if (std::find(orders.begin(), orders.end(), 3) != orders.end()) {
    en3_1pCore.resize(electron_pairs * electron_pairs);
    en3_2pCore.resize(electron_pairs * electron_pairs);
    en3_12pCore.resize(electron_pairs * electron_pairs);
    en3_1mCore.resize(electron_pairs * electron_pairs);
    en3_2mCore.resize(electron_pairs * electron_pairs);
    en3_12mCore.resize(electron_pairs * electron_pairs);
    en3_12cCore.resize(electron_pairs * electron_pairs);
    en3_22cCore.resize(electron_pairs * electron_pairs);
    en3c12.resize(electron_pairs);
    en3c22.resize(electron_pairs);
    one.resize(electron_pairs, 1.0);
  }
  if (numDiff > 1) {
    auto max_order = *std::max_element(orders.begin(), orders.end());
    if (max_order == 2) {
      enGrouped.resize(nmo * nmo * 2);
      en2p = enGrouped.data() + 0 * nmo * nmo;
      en2m = enGrouped.data() + 1 * nmo * nmo;
    } else if (max_order == 3) {
      enGrouped.resize(nmo * nmo * 7);
      en2p = enGrouped.data() + 0 * nmo * nmo;
      en2m = enGrouped.data() + 1 * nmo * nmo;

      en3_1p  = enGrouped.data() + 0 * nmo * nmo;
      en3_1m  = enGrouped.data() + 1 * nmo * nmo;
      en3_2p  = enGrouped.data() + 2 * nmo * nmo;
      en3_2m  = enGrouped.data() + 3 * nmo * nmo;
      en3_12p = enGrouped.data() + 4 * nmo * nmo;
      en3_12m = enGrouped.data() + 5 * nmo * nmo;
      en3_c   = enGrouped.data() + 6 * nmo * nmo;
    } else {
      exit(0);
    }
  }
}

OVPS_ARRAY::OVPS_ARRAY(const OVPS_ARRAY& other) {
  exit(0);
}

OVPS_ARRAY& OVPS_ARRAY::operator = (const OVPS_ARRAY& other) {
  std::cerr << "GET FUCKED\n";
  exit(0);
}

void OVPS_ARRAY::zero_energy_arrays() {
  for (auto& it : enBlock) {
    for (auto& jt : it) {
      std::fill(jt.begin(), jt.end(), 0.0);
    }
  }
}

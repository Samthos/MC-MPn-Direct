  TIDX_CONTROL {
    TIDY_CONTROL {
      double en3 = 0;
      int index = tidy * mc_pair_num + tidx;
      int jkIndex, ijIndex, ikIndex;

      if(tidx != tidy) {
        double en[10];
        for(int tidz = 0; tidz < 10; tidz++) {
          en[tidz] = 0;
        }
        for(int tidz = 0; tidz < mc_pair_num; tidz++) {
          if(tidx != tidz && tidy != tidz) {
            ijIndex = tidz * mc_pair_num + tidx;
            ikIndex = tidz * mc_pair_num + tidy;
            en[0] += ovps.os_13[ijIndex] * ovps.os_24[ijIndex] * ovps.vs_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.rv[tidz];  //
            en[1] += ovps.os_13[ijIndex] * ovps.os_24[ijIndex] * ovps.vs_16[ikIndex] * ovps.vs_25[ikIndex] * ovps.rv[tidz];  //
            en[2] += ovps.os_24[ijIndex] * ovps.vs_23[ijIndex] * ovps.os_15[ikIndex] * ovps.vs_16[ikIndex] * ovps.rv[tidz];  //
            en[3] += ovps.os_23[ijIndex] * ovps.vs_23[ijIndex] * ovps.os_15[ikIndex] * ovps.vs_16[ikIndex] * ovps.rv[tidz];  // 34
            en[4] += ovps.os_24[ijIndex] * ovps.vs_13[ijIndex] * ovps.os_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.rv[tidz];  //
            en[5] += ovps.os_23[ijIndex] * ovps.vs_13[ijIndex] * ovps.os_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.rv[tidz];  // 34
            en[6] += ovps.os_24[ijIndex] * ovps.vs_23[ijIndex] * ovps.os_15[ikIndex] * ovps.vs_15[ikIndex] * ovps.rv[tidz];  //
            en[7] += ovps.os_23[ijIndex] * ovps.vs_23[ijIndex] * ovps.os_15[ikIndex] * ovps.vs_15[ikIndex] * ovps.rv[tidz];  // 34
            en[8] += ovps.os_24[ijIndex] * ovps.vs_13[ijIndex] * ovps.os_15[ikIndex] * ovps.vs_25[ikIndex] * ovps.rv[tidz];  //
            en[9] += ovps.os_23[ijIndex] * ovps.vs_13[ijIndex] * ovps.os_15[ikIndex] * ovps.vs_25[ikIndex] * ovps.rv[tidz];  // 34
          }
        }
        jkIndex = tidx * mc_pair_num + tidy;
        en3 +=  2.00 * en[0] * ovps.os_35[jkIndex];  //
        en3 += -1.00 * en[1] * ovps.os_35[jkIndex];  //
        en3 +=  1.00 * en[2] * ovps.vs_35[jkIndex];  //
        en3 += -2.00 * en[3] * ovps.vs_45[jkIndex];  // 34
        en3 += -2.00 * en[4] * ovps.vs_35[jkIndex];  //
        en3 +=  1.00 * en[5] * ovps.vs_45[jkIndex];  // 34
        en3 += -2.00 * en[6] * ovps.vs_36[jkIndex];  //
        en3 +=  4.00 * en[7] * ovps.vs_46[jkIndex];  // 34
        en3 +=  1.00 * en[8] * ovps.vs_36[jkIndex];  //
        en3 += -2.00 * en[9] * ovps.vs_46[jkIndex];  // 34
        en3 = en3 * ovps.rv[tidx] * ovps.rv[tidy];
      }
      ovps.en3_2pCore[index] = en3;
    }
  }

  TIDX_CONTROL {
    TIDY_CONTROL {
      double en3_12 = 0;
      double en3_22 = 0;

      int index = tidy * mc_pair_num + tidx;
      int ijIndex, ikIndex, jkIndex;

      if(tidx != tidy) {
        double en[4];
        for(int tidz = 0; tidz < 4; tidz++) {
          en[tidz] = 0;
        }
        for(int tidz = 0; tidz < mc_pair_num; tidz++) {
          if(tidx != tidz && tidy != tidz) {
            ikIndex = tidx * mc_pair_num + tidz;
            jkIndex = tidy * mc_pair_num + tidz;
            double temp = ovps.os_35[jkIndex] * ovps.os_46[jkIndex] * ovps.rv[tidz];
            en[0] += temp * ovps.vs_35[jkIndex] * ovps.vs_16[ikIndex];  // 12
            en[1] += temp * ovps.vs_36[jkIndex] * ovps.vs_15[ikIndex];  // 12

            temp = ovps.os_35[jkIndex] * ovps.os_16[ikIndex] * ovps.rv[tidz];
            en[2] += temp * ovps.vs_35[jkIndex] * ovps.vs_46[jkIndex];
            en[3] += temp * ovps.vs_36[jkIndex] * ovps.vs_45[jkIndex];
          }
        }
        ijIndex = tidx * mc_pair_num + tidy;
        en3_12 +=  2.00 * en[0] * ovps.os_24[ijIndex];  // * ovps.ps_12c[ibIndex];  // 12
        en3_12 += -1.00 * en[1] * ovps.os_24[ijIndex];  // * ovps.ps_12c[ibIndex];  // 12
        en3_12 += -2.00 * en[2] * ovps.vs_24[ijIndex];  // * ovps.ps_12c[ibIndex];
        en3_12 +=  1.00 * en[3] * ovps.vs_24[ijIndex];  // * ovps.ps_12c[ibIndex];

        en3_22 += -4.00 * en[0] * ovps.os_14[ijIndex];  // * ovps.ps_22c[ibIndex];
        en3_22 +=  2.00 * en[1] * ovps.os_14[ijIndex];  // * ovps.ps_22c[ibIndex];
        en3_22 +=  4.00 * en[2] * ovps.vs_14[ijIndex];  // * ovps.ps_22c[ibIndex];
        en3_22 += -2.00 * en[3] * ovps.vs_14[ijIndex];  // * ovps.ps_22c[ibIndex];

        for(int tidz = 0; tidz < 4; tidz++) {
          en[tidz] = 0;
        }
        for(int tidz = 0; tidz < mc_pair_num; tidz++) {
          if(tidx != tidz && tidy != tidz) {
            ikIndex = tidy * mc_pair_num + tidz;
            jkIndex = tidx * mc_pair_num + tidz;
            double temp = ovps.os_35[jkIndex] * ovps.os_16[ikIndex] * ovps.rv[tidz];
            en[0] += temp * ovps.vs_15[ikIndex] * ovps.vs_26[ikIndex];
            en[1] += temp * ovps.vs_16[ikIndex] * ovps.vs_25[ikIndex];

            temp = ovps.os_15[ikIndex] * ovps.os_26[ikIndex] * ovps.vs_35[jkIndex] * ovps.rv[tidz];
            en[2] += temp * ovps.vs_16[ikIndex];
            en[3] += temp * ovps.vs_26[ikIndex];
          }
        }
        ijIndex = tidy * mc_pair_num + tidx;
        en3_12 += -1.00 * en[0] * ovps.os_24[ijIndex];  // * ovps.ps_34c[jbIndex];
        en3_12 +=  2.00 * en[1] * ovps.os_24[ijIndex];  // * ovps.ps_34c[jbIndex];
        en3_12 +=  1.00 * en[2] * ovps.vs_24[ijIndex];  // * ovps.ps_34c[jbIndex];
        en3_12 += -2.00 * en[3] * ovps.vs_14[ijIndex];  // * ovps.ps_34c[jbIndex];
                                
        en3_22 +=  2.00 * en[0] * ovps.os_23[ijIndex];  // * ovps.ps_44c[jbIndex];
        en3_22 += -4.00 * en[1] * ovps.os_23[ijIndex];  // * ovps.ps_44c[jbIndex];
        en3_22 += -2.00 * en[2] * ovps.vs_23[ijIndex];  // * ovps.ps_44c[jbIndex];
        en3_22 +=  4.00 * en[3] * ovps.vs_13[ijIndex];  // * ovps.ps_44c[jbIndex];

        for(int tidz = 0; tidz < 4; tidz++) {
          en[tidz] = 0;
        }
        for(int tidz = 0; tidz < mc_pair_num; tidz++) {
          if(tidx != tidz && tidy != tidz) {
            ijIndex = tidy * mc_pair_num + tidz;
            jkIndex = tidz * mc_pair_num + tidx;
            double temp = ovps.os_13[ijIndex] * ovps.os_35[jkIndex] * ovps.os_24[ijIndex] * ovps.rv[tidz];
            en[0] += temp * ovps.vs_24[ijIndex];
            en[1] += temp * ovps.vs_14[ijIndex];

            temp = ovps.os_24[ijIndex] * ovps.vs_35[jkIndex] * ovps.rv[tidz];
            en[2] += temp * ovps.vs_13[ijIndex] * ovps.vs_24[ijIndex];  // 56
            en[3] += temp * ovps.vs_14[ijIndex] * ovps.vs_23[ijIndex];  // 56
          }
        }
        ikIndex = tidy * mc_pair_num + tidx;
        en3_12 +=  2.00 * en[0] * ovps.vs_16[ikIndex];  // * ovps.ps_56c[kbIndex];
        en3_12 += -1.00 * en[1] * ovps.vs_26[ikIndex];  // * ovps.ps_56c[kbIndex];
        en3_12 += -2.00 * en[2] * ovps.os_16[ikIndex];  // * ovps.ps_56c[kbIndex]; // 56
        en3_12 +=  1.00 * en[3] * ovps.os_16[ikIndex];  // * ovps.ps_56c[kbIndex]; // 56
                                
        en3_22 += -4.00 * en[0] * ovps.vs_15[ikIndex];  // * ovps.ps_66c[kbIndex];
        en3_22 +=  2.00 * en[1] * ovps.vs_25[ikIndex];  // * ovps.ps_66c[kbIndex];
        en3_22 +=  4.00 * en[2] * ovps.os_15[ikIndex];  // * ovps.ps_66c[kbIndex];
        en3_22 += -2.00 * en[3] * ovps.os_15[ikIndex];  // * ovps.ps_66c[kbIndex];

        en3_12 *= ovps.rv[tidx] * ovps.rv[tidy];
        en3_22 *= ovps.rv[tidx] * ovps.rv[tidy];
      }
      ovps.en3_12cCore[index] = en3_12;
      ovps.en3_22cCore[index] = en3_22;
    }
  }

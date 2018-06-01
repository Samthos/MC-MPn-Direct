TIDX_CONTROL {
  TIDY_CONTROL {
    double en3 = 0;
    int index = tidy * mc_pair_num + tidx;
    int jkIndex, ijIndex, ikIndex;

    if (tidx != tidy) {
      double en[10];
      for (int tidz = 0; tidz < 10; tidz++) {
        en[tidz] = 0;
      }
      for (int tidz = 0; tidz < mc_pair_num; tidz++) {
        if (tidx != tidz && tidy != tidz) {
          ijIndex = tidx * mc_pair_num + tidz;
          jkIndex = tidz * mc_pair_num + tidy;
          en[0] += ovps.os_13[ijIndex] * ovps.os_24[ijIndex] * ovps.os_35[jkIndex] * ovps.os_46[jkIndex] * ovps.rv[tidz];  //
          en[1] += ovps.os_13[ijIndex] * ovps.os_24[ijIndex] * ovps.os_36[jkIndex] * ovps.os_45[jkIndex] * ovps.rv[tidz];  // 56
          en[2] += ovps.os_24[ijIndex] * ovps.vs_14[ijIndex] * ovps.os_35[jkIndex] * ovps.vs_35[jkIndex] * ovps.rv[tidz];  //
          en[3] += ovps.os_14[ijIndex] * ovps.vs_14[ijIndex] * ovps.os_35[jkIndex] * ovps.vs_35[jkIndex] * ovps.rv[tidz];  // 12
          en[4] += ovps.os_24[ijIndex] * ovps.vs_14[ijIndex] * ovps.os_36[jkIndex] * ovps.vs_35[jkIndex] * ovps.rv[tidz];  // 56
          en[5] += ovps.os_14[ijIndex] * ovps.vs_14[ijIndex] * ovps.os_36[jkIndex] * ovps.vs_35[jkIndex] * ovps.rv[tidz];  // 12 56
          en[6] += ovps.os_23[ijIndex] * ovps.vs_14[ijIndex] * ovps.os_35[jkIndex] * ovps.vs_45[jkIndex] * ovps.rv[tidz];  //
          en[7] += ovps.os_13[ijIndex] * ovps.vs_14[ijIndex] * ovps.os_35[jkIndex] * ovps.vs_45[jkIndex] * ovps.rv[tidz];  // 12
          en[8] += ovps.os_23[ijIndex] * ovps.vs_14[ijIndex] * ovps.os_36[jkIndex] * ovps.vs_45[jkIndex] * ovps.rv[tidz];  // 56
          en[9] += ovps.os_13[ijIndex] * ovps.vs_14[ijIndex] * ovps.os_36[jkIndex] * ovps.vs_45[jkIndex] * ovps.rv[tidz];  // 12 56
        }
      }
      ikIndex = tidx * mc_pair_num + tidy;
      en3 += -2.00 * en[0] * ovps.vs_15[ikIndex];  //
      en3 += 1.00 * en[1] * ovps.vs_15[ikIndex];   // 56
      en3 += 2.00 * en[2] * ovps.os_16[ikIndex];   //
      en3 += -4.00 * en[3] * ovps.os_26[ikIndex];  // 12
      en3 += -1.00 * en[4] * ovps.os_15[ikIndex];  // 56
      en3 += 2.00 * en[5] * ovps.os_25[ikIndex];   // 12 56
      en3 += -1.00 * en[6] * ovps.os_16[ikIndex];  //
      en3 += 2.00 * en[7] * ovps.os_26[ikIndex];   // 12
      en3 += 2.00 * en[8] * ovps.os_15[ikIndex];   // 56
      en3 += -1.00 * en[9] * ovps.os_25[ikIndex];  // 12 56
      en3 = en3 * ovps.rv[tidx] * ovps.rv[tidy];
    }
    ovps.en3_12mCore[index] = en3;
  }
}

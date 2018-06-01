TIDX_CONTROL {
  TIDY_CONTROL {
    int Index = tidy * mc_pair_num + tidx;
    double en2m, en2p;
    en2m = 0;
    en2p = 0;
    if(tidx != tidy) {
      en2p = en2p - 2.00 * ovps.os_13[Index] * ovps.vs_24[Index] * ovps.vs_13[Index];  // ovps.ps_24[bandIndex];
      en2p = en2p + 1.00 * ovps.os_13[Index] * ovps.vs_14[Index] * ovps.vs_23[Index];  // ovps.ps_24[bandIndex];

      en2m = en2m + 2.00 * ovps.os_24[Index] * ovps.os_13[Index] * ovps.vs_13[Index];  // ovps.ps_24c[bandIndex];
      en2m = en2m - 1.00 * ovps.os_23[Index] * ovps.os_14[Index] * ovps.vs_13[Index];  // ovps.ps_24c[bandIndex];

      en2p = en2p * (ovps.rv[tidx] * ovps.rv[tidy]);
      en2m = en2m * (ovps.rv[tidx] * ovps.rv[tidy]);
    }
    ovps.en2pCore[Index] = en2p;
    ovps.en2mCore[Index] = en2m;
  }
}

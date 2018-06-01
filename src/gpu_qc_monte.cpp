#include "qc_monte.h"

void QC_monte::update_wavefunction() {
  double pos[iops.iopns[KEYS::MC_NPAIR]*6];

  for(int i=0;i<iops.iopns[KEYS::MC_NPAIR];i++) {
    for(int j=0;j<3;j++) {
      pos[i*3+j] = el_pair_list[i].pos1[j] ;
      pos[(i+iops.iopns[KEYS::MC_NPAIR])*3+j] = el_pair_list[i].pos2[j];
    }
  }
  basis.device_psi_get(ovps.d_ovps.occ1, ovps.d_ovps.occ2, ovps.d_ovps.vir1, ovps.d_ovps.vir2, ovps.d_ovps.psi1, ovps.d_ovps.psi2, pos, iops.iopns[KEYS::MC_NPAIR]);
}
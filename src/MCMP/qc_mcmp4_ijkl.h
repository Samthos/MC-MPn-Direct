// double il
en[0]  = ovps.o_set[2][0].s_11[il] * ovps.o_set[2][0].s_22[il] * ovps.v_set[2][1].s_12[jl] * ovps.v_set[2][2].s_11[kl];

en[1]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_11[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][2].s_12[kl];
en[5]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_11[il] * ovps.v_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl];
en[3]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_21[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][2].s_12[kl];
en[9]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_21[il] * ovps.v_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl];
en[2]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_12[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][2].s_11[kl];
en[7]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_12[il] * ovps.v_set[2][1].s_11[jl] * ovps.o_set[2][2].s_11[kl];
en[4]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_22[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][2].s_11[kl];
en[11] = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_22[il] * ovps.v_set[2][1].s_11[jl] * ovps.o_set[2][2].s_11[kl];

en[13] = ovps.v_set[2][0].s_11[il] * ovps.v_set[2][0].s_22[il] * ovps.o_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl];


// double jl
en[14] = ovps.o_set[2][1].s_11[jl] * ovps.o_set[2][1].s_22[jl] * ovps.v_set[2][0].s_12[il] * ovps.v_set[2][2].s_11[kl];

en[15]  = ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][1].s_11[jl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][2].s_12[kl];
en[21] = ovps.o_set[2][1].s_12[jl] * ovps.v_set[2][1].s_11[jl] * ovps.v_set[2][0].s_12[il] * ovps.o_set[2][2].s_11[kl];
en[17]  = ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][1].s_21[jl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][2].s_12[kl];
en[22] = ovps.o_set[2][1].s_12[jl] * ovps.v_set[2][1].s_21[jl] * ovps.v_set[2][0].s_12[il] * ovps.o_set[2][2].s_11[kl];
en[16]  = ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][1].s_12[jl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][2].s_11[kl];
en[19] = ovps.o_set[2][1].s_12[jl] * ovps.v_set[2][1].s_12[jl] * ovps.v_set[2][0].s_11[il] * ovps.o_set[2][2].s_11[kl];
en[18]  = ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][1].s_22[jl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][2].s_11[kl];
en[20] = ovps.o_set[2][1].s_12[jl] * ovps.v_set[2][1].s_22[jl] * ovps.v_set[2][0].s_11[il] * ovps.o_set[2][2].s_11[kl];

en[27] = ovps.v_set[2][1].s_11[jl] * ovps.v_set[2][1].s_22[jl] * ovps.o_set[2][0].s_12[il] * ovps.o_set[2][2].s_11[kl];

// double kl
en[28] = ovps.o_set[2][2].s_11[kl] * ovps.o_set[2][2].s_22[kl] * ovps.v_set[2][0].s_11[il] * ovps.v_set[2][1].s_12[jl];

en[36] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_12[kl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_11[jl];
en[37] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_22[kl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_11[jl];
en[38] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_11[kl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_12[jl];
en[39] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_21[kl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_12[jl];
en[40] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_12[kl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_21[jl];
en[41] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_22[kl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_21[jl];
en[42] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_11[kl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_22[jl];
en[43] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_21[kl] * ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_22[jl];
en[44] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_12[kl] * ovps.v_set[2][0].s_11[il] * ovps.o_set[2][1].s_12[jl];
en[45] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_22[kl] * ovps.v_set[2][0].s_11[il] * ovps.o_set[2][1].s_12[jl];
en[46] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_11[kl] * ovps.v_set[2][0].s_12[il] * ovps.o_set[2][1].s_12[jl];
en[47] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_21[kl] * ovps.v_set[2][0].s_12[il] * ovps.o_set[2][1].s_12[jl];
en[48] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_12[kl] * ovps.v_set[2][0].s_21[il] * ovps.o_set[2][1].s_12[jl];
en[49] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_22[kl] * ovps.v_set[2][0].s_21[il] * ovps.o_set[2][1].s_12[jl];
en[50] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_11[kl] * ovps.v_set[2][0].s_22[il] * ovps.o_set[2][1].s_12[jl];
en[51] = ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_21[kl] * ovps.v_set[2][0].s_22[il] * ovps.o_set[2][1].s_12[jl];

en[52] = ovps.v_set[2][2].s_11[kl] * ovps.v_set[2][2].s_22[kl] * ovps.o_set[2][0].s_12[il] * ovps.o_set[2][1].s_11[jl];

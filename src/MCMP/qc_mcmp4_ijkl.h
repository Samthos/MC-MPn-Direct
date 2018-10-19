// double il
en[0]  = ovps.o_set[2][0].s_11[il] * ovps.o_set[2][0].s_22[il] * ovps.v_set[2][1].s_12[jl] * ovps.v_set[2][2].s_11[kl];
en[1]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_11[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][2].s_12[kl];
en[2]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_11[il] * ovps.v_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl];
en[3]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_21[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][2].s_12[kl];
en[4]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_21[il] * ovps.v_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl];
en[5]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_12[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][2].s_11[kl];
en[6]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_12[il] * ovps.v_set[2][1].s_11[jl] * ovps.o_set[2][2].s_11[kl];
en[7]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_22[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][2].s_11[kl];
en[8]  = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][0].s_22[il] * ovps.v_set[2][1].s_11[jl] * ovps.o_set[2][2].s_11[kl];
en[9]  = ovps.v_set[2][0].s_11[il] * ovps.v_set[2][0].s_22[il] * ovps.o_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl];


// double jl
en[10] = ovps.v_set[2][0].s_12[il] * ovps.o_set[2][1].s_11[jl] * ovps.o_set[2][1].s_22[jl] * ovps.v_set[2][2].s_11[kl];
en[11] = ovps.o_set[2][0].s_12[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][1].s_11[jl] * ovps.v_set[2][2].s_12[kl];
en[12] = ovps.v_set[2][0].s_12[il] * ovps.o_set[2][1].s_12[jl] * ovps.v_set[2][1].s_11[jl] * ovps.o_set[2][2].s_11[kl];
en[13] = ovps.o_set[2][0].s_12[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][1].s_21[jl] * ovps.v_set[2][2].s_12[kl];
en[14] = ovps.v_set[2][0].s_12[il] * ovps.o_set[2][1].s_12[jl] * ovps.v_set[2][1].s_21[jl] * ovps.o_set[2][2].s_11[kl];
en[15] = ovps.o_set[2][0].s_12[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][1].s_12[jl] * ovps.v_set[2][2].s_11[kl];
en[16] = ovps.v_set[2][0].s_11[il] * ovps.o_set[2][1].s_12[jl] * ovps.v_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl];
en[17] = ovps.o_set[2][0].s_12[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][1].s_22[jl] * ovps.v_set[2][2].s_11[kl];
en[18] = ovps.v_set[2][0].s_11[il] * ovps.o_set[2][1].s_12[jl] * ovps.v_set[2][1].s_22[jl] * ovps.o_set[2][2].s_11[kl];
en[19] = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_11[jl] * ovps.v_set[2][1].s_22[jl] * ovps.o_set[2][2].s_11[kl];


// double kl
en[20] = ovps.v_set[2][0].s_11[il] * ovps.v_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl] * ovps.o_set[2][2].s_22[kl];
en[21] = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_11[kl];
en[22] = ovps.v_set[2][0].s_12[il] * ovps.o_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_11[kl];
en[23] = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_21[kl];
en[24] = ovps.v_set[2][0].s_12[il] * ovps.o_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_21[kl];
en[25] = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_11[jl] * ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_12[kl];
en[26] = ovps.v_set[2][0].s_11[il] * ovps.o_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_12[kl];
en[27] = ovps.o_set[2][0].s_12[il] * ovps.v_set[2][1].s_11[jl] * ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_22[kl];
en[28] = ovps.v_set[2][0].s_11[il] * ovps.o_set[2][1].s_12[jl] * ovps.o_set[2][2].s_11[kl] * ovps.v_set[2][2].s_22[kl];
en[29] = ovps.o_set[2][0].s_12[il] * ovps.o_set[2][1].s_11[jl] * ovps.v_set[2][2].s_11[kl] * ovps.v_set[2][2].s_22[kl];

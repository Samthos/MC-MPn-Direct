en[0 ] = 1 * ovps.o_set[0][0].s_11[it_jt] * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.o_set[1][1].s_22[jt_kt];
en[1 ] = 1 * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[0][0].s_21[it_jt] * ovps.v_set[1][1].s_11[jt_kt] * ovps.v_set[1][1].s_22[jt_kt];
en[2 ] = 2 * ovps.o_set[0][0].s_21[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[1][1].s_22[jt_kt];
en[3 ] = 2 * ovps.o_set[0][0].s_21[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_22[jt_kt];
en[4 ] = 2 * ovps.o_set[0][0].s_21[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[1][1].s_22[jt_kt];
en[5 ] = 2 * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[6 ] = 4 * ovps.o_set[0][0].s_11[it_jt] * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_12[it_jt];
en[7 ] = 4 * ovps.o_set[0][0].s_11[it_jt] * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_22[it_jt];
en[8 ] = 4 * ovps.o_set[0][0].s_21[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.o_set[1][1].s_22[jt_kt] * ovps.v_set[1][1].s_22[jt_kt];
en[9 ] = 4 * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.o_set[1][1].s_22[jt_kt] * ovps.v_set[1][1].s_12[jt_kt];
en[10] = 4 * ovps.o_set[0][0].s_22[it_jt] * ovps.v_set[0][0].s_11[it_jt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[11] = 4 * ovps.o_set[0][0].s_22[it_jt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[0][0].s_21[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[12] = 4 * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[1][1].s_11[jt_kt] * ovps.v_set[1][1].s_22[jt_kt];
en[13] = 4 * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_12[jt_kt] * ovps.v_set[1][1].s_21[jt_kt];
en[14] = 8 * ovps.o_set[0][0].s_21[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_22[jt_kt];
en[15] = 8 * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[16] = 8 * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[17] = 8 * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[18] = -2 * ovps.o_set[0][0].s_11[it_jt] * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.o_set[1][1].s_22[jt_kt];
en[19] = -2 * ovps.o_set[0][0].s_11[it_jt] * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_12[it_jt];
en[20] = -2 * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.o_set[1][1].s_22[jt_kt] * ovps.v_set[1][1].s_12[jt_kt];
en[21] = -2 * ovps.o_set[0][0].s_22[it_jt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[0][0].s_21[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[22] = -2 * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_11[jt_kt] * ovps.v_set[1][1].s_22[jt_kt];
en[23] = -2 * ovps.v_set[0][0].s_11[it_jt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_11[jt_kt] * ovps.v_set[1][1].s_22[jt_kt];
en[24] = -4 * ovps.o_set[0][0].s_21[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[1][1].s_22[jt_kt];
en[25] = -4 * ovps.o_set[0][0].s_21[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_22[jt_kt];
en[26] = -4 * ovps.o_set[0][0].s_21[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[1][1].s_22[jt_kt];
en[27] = -4 * ovps.o_set[0][0].s_21[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_22[jt_kt];
en[28] = -4 * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[29] = -4 * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[30] = -4 * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[31] = -8 * ovps.o_set[0][0].s_11[it_jt] * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_22[it_jt];
en[32] = -8 * ovps.o_set[0][0].s_21[it_jt] * ovps.o_set[1][1].s_11[jt_kt] * ovps.o_set[1][1].s_22[jt_kt] * ovps.v_set[1][1].s_22[jt_kt];
en[33] = -8 * ovps.o_set[0][0].s_22[it_jt] * ovps.v_set[0][0].s_11[it_jt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_12[jt_kt];
en[34] = -8 * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_12[it_jt] * ovps.v_set[1][1].s_12[jt_kt] * ovps.v_set[1][1].s_21[jt_kt];
en[35] = -16 * ovps.o_set[0][0].s_22[it_jt] * ovps.o_set[1][1].s_12[jt_kt] * ovps.v_set[0][0].s_22[it_jt] * ovps.v_set[1][1].s_12[jt_kt];

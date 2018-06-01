#include <algorithm>

#include "el_pair.h"
#include "qc_constant.h"

void el_pair_typ::init(int ivir2) {
	psi1.resize(ivir2);
	psi2.resize(ivir2);
}

void el_pair_typ::pos_init(Molec& molec, Random& rand){
	int i;
	double amp1, amp2, theta1, theta2;
	double pos[3];

	//elec position 1
	i = molec.natom * rand.get_rand();
	pos[0] = molec.atom[i].pos[0];
	pos[1] = molec.atom[i].pos[1];
	pos[2] = molec.atom[i].pos[2];

	amp1 = sqrt(-0.5 * log(rand.get_rand() * 0.2));
	amp2 = sqrt(-0.5 * log(rand.get_rand() * 0.5));
	theta1 = twopi * rand.get_rand();
	theta2 = pi * rand.get_rand();

	pos1[0] = pos[0] + amp1*cos(theta1);
	pos1[1] = pos[1] + amp1*sin(theta1);
	pos1[2] = pos[2] + amp2*cos(theta2);

	//elec position 2;
	i = molec.natom * rand.get_rand();
	pos[0] = molec.atom[i].pos[0];
	pos[1] = molec.atom[i].pos[1];
	pos[2] = molec.atom[i].pos[2];

	amp1 = sqrt(-0.5 * log(rand.get_rand() * 0.2));
	amp2 = sqrt(-0.5 * log(rand.get_rand() * 0.5));
	theta1 = twopi * rand.get_rand();
	theta2 = pi * rand.get_rand();

	pos2[0] = pos[0] + amp1*cos(theta1);
	pos2[1] = pos[1] + amp1*sin(theta1);
	pos2[2] = pos[2] + amp2*cos(theta2);
}

void el_pair_typ::weight_func_set(Molec& molec, MC_Basis& mc_basis){
	int i,j,k;
	double r1, r2, dr;
	double gf1, gf2, azi, normi;


	r12 = 0.00;
	for(i=0;i<3;i++) {
		dr = pos1[i] - pos2[i];
		r12 = r12 + dr*dr;
	}
	r12 = sqrt(r12);

	gf1 = 0.0;
	gf2 = 0.0;
	for(i=0;i<molec.natom;i++) {
		k = mc_basis.atom_ibas[i];

		r1 = 0.00;
		r2 = 0.00;
		for(j=0;j<3;j++) {
			dr = pos1[j] - molec.atom[i].pos[j];
			r1 = r1 + dr*dr;
	
			dr = pos2[j] - molec.atom[i].pos[j];
			r2 = r2 + dr*dr;
		}
	
		for(j=0;j<mc_basis.mc_nprim;j++) {
			azi   = mc_basis.mc_basis_list[k].alpha[j];
			normi = mc_basis.mc_basis_list[k].norm[j];
			
			gf1 = gf1 + exp(-azi*r1)*normi;
			gf2 = gf2 + exp(-azi*r2)*normi;
		}
	}

	wgt = gf1*gf2/r12;
	rv = 1.0/(gf1*gf2);
//	std::cerr << r12 << "\t" << wgt << std::endl;
}

void el_pair_typ::mc_move_scheme(int *nsucc,
	 	int *nfail,
	 	double delx,
		Random& rand,
		Molec& molec,
		MC_Basis& mc_basis) {
	double pos1_old[3], pos2_old[3], wgt_old, r12_old, rv_old;
	double ratio, rnd3[3], rval;

	pos1_old[0] = pos1[0];
	pos1_old[1] = pos1[1];
	pos1_old[2] = pos1[2];
	pos2_old[0] = pos2[0];
	pos2_old[1] = pos2[1];
	pos2_old[2] = pos2[2];
	wgt_old  = wgt;
	r12_old  = r12;
	rv_old   = rv;

	rand.get_rand3(rnd3);
	pos1[0] = pos1_old[0] + (rnd3[0] - 0.5)*delx;
	pos1[1] = pos1_old[1] + (rnd3[1] - 0.5)*delx;
	pos1[2] = pos1_old[2] + (rnd3[2] - 0.5)*delx;

	rand.get_rand3(rnd3);
	pos2[0] = pos2_old[0] + (rnd3[0] - 0.5)*delx;
	pos2[1] = pos2_old[1] + (rnd3[1] - 0.5)*delx;
	pos2[2] = pos2_old[2] + (rnd3[2] - 0.5)*delx;

	weight_func_set(molec, mc_basis);

	ratio = wgt/wgt_old;
	rval = rand.get_rand();

	if (rval < 1.0E-3) {
		rval = 1.0E-3;
	}

	is_new = true;
	if (ratio > rval) {
		*nsucc = *nsucc + 1;
	}
	else {
		*nfail = *nfail + 1;
		pos1[0] = pos1_old[0];
		pos1[1] = pos1_old[1];
		pos1[2] = pos1_old[2];
		pos2[0] = pos2_old[0];
		pos2[1] = pos2_old[1];
		pos2[2] = pos2_old[2];
		wgt  = wgt_old;
		r12  = r12_old;
		rv  = rv_old;
		is_new = false;
	}
//	std::cerr << pos1[0] << "\t" << pos1[1] << "\t" << pos1[2] << "\n";
//	std::cerr << pos2[0] << "\t" << pos2[1] << "\t" << pos2[2] << "\n";
}

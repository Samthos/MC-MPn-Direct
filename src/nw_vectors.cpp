#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "qc_basis.h"
#include "qc_geom.h"
#include "mpi.h"

void Basis::nw_vectors_read(MPI_info& mpi_info, Molec& molec, IOPs& iops) {
	int i,j;
	long long titleLength;
	long long basisTitleLength;
	int ignoreInt;
	char ignoreChar[256];
	double *occ;
	std::string scftype20;
	std::string title;
	std::string basis_name;

	std::ifstream input;

	int readmode = iops.iopns[KEYS::MOVECS];
	if(mpi_info.sys_master) {
		if(readmode == 0) {
			std::cout << "Reading binary MOVECS from " << iops.sopns[KEYS::MOVECS] << std::endl;
			input.open(iops.sopns[KEYS::MOVECS].c_str(), std::ios::binary);
			if(!input.is_open()) {
				std::cerr << "no movecs file" << std::endl;
				exit(EXIT_FAILURE);
			}

			//get calcaultion info
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n", ignoreInt); //debug
			input.read(ignoreChar,ignoreInt);
			ignoreChar[ignoreInt] = '\0';
//			std::cout << ignoreChar << std::endl; //debug
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n\n", ignoreInt); //debug

			//calcualtion type
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n", ignoreInt); //debug
			input.read(ignoreChar,ignoreInt);
			ignoreChar[ignoreInt] = '\0';
//			std::cout << ignoreChar << std::endl; //debug
			input.read((char*) &ignoreInt,4); 
//			printf("%#08x\n\n", ignoreInt);//debug

			//title length
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n", ignoreInt); //debug
			input.read((char*) &titleLength,ignoreInt);
//			std::cout << "title Length: " << titleLength << std::endl; //debug
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n\n", ignoreInt); //debug

			//title
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n", ignoreInt); //debug
			input.read(ignoreChar,ignoreInt);
			ignoreChar[ignoreInt] = '\0';
//			std::cout << ignoreChar << std::endl; //debug
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n\n", ignoreInt); //debug

			//basis name length
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n", ignoreInt); //debug
			input.read((char*) &basisTitleLength,ignoreInt);
//			std::cout << "basis title Length: " << basisTitleLength << std::endl; //debug
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n\n", ignoreInt); //debug

			//basis name
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n", ignoreInt); //debug
			input.read(ignoreChar,ignoreInt);
			ignoreChar[ignoreInt] = '\0';
//			std::cout << ignoreChar << std::endl; //debug
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n\n", ignoreInt); //debug

			//nwsets
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n", ignoreInt); //debug
			input.read((char*) &nw_nsets,ignoreInt);
//			std::cout << "nw sets: " << nw_nsets << std::endl; //debug
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n\n", ignoreInt); //debug

			//nw_nbf
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n", ignoreInt); //debug
			input.read((char*) &nw_nbf,ignoreInt);
//			std::cout << "nbf: " << nw_nbf << std::endl; //debug
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n\n", ignoreInt); //debug

			//nw_nmo
			if(nw_nsets > 1) {
				std::cerr << "nw_nsets > 1" << std::endl;
				std::cerr << "Code only supports nw_nset==1" << std::endl;
				std::cerr << "Please contact Alex" << std::endl;
				exit(EXIT_FAILURE);
			}
			else {
				input.read((char*) &ignoreInt,4);
//				printf("%#08x\n", ignoreInt); //debug
				input.read((char*) nw_nmo,ignoreInt);
//				std::cout << "nw_nmo: " << nw_nmo[0] << std::endl; //debug
				input.read((char*) &ignoreInt,4);
//				printf("%#08x\n\n", ignoreInt); //debug
			}

		}
		else{
			std::cout << "Reading ascii MOVECS from " << iops.sopns[KEYS::MOVECS] << std::endl;
			input.open(iops.sopns[KEYS::MOVECS].c_str());
			if(!input.is_open()) {
				std::cerr << "no movecs file" << std::endl;
				exit(EXIT_FAILURE);
			}
			input.ignore(1000,'\n'); // #
			input.ignore(1000,'\n'); // skip convergence info
			input.ignore(1000,'\n'); // skip convergence info
			input.ignore(1000,'\n'); // space
			input.ignore(1000,'\n'); // scftype20
			input.ignore(1000,'\n'); // date lentit
			input >> scftype20;
			input >> titleLength;
			input.ignore(1000,'\n');
			std::getline(input,title);


			input >> basisTitleLength;
			input.ignore(1000,'\n');
			std::getline(input,basis_name);

			input >> nw_nsets >> nw_nbf;

			nw_nmo[0] = 0;
			nw_nmo[1] = 0;
			
			for(i=0;i<nw_nsets;i++) {
				input >> nw_nmo[i];
			}

		}
		std::cout << "nw_vectors: nbf " <<  nw_nbf << std::endl;
		std::cout << "nw_vectors: nmo ";
		for(i=0;i<nw_nsets;i++) {
			std::cout << nw_nmo[i] << " ";
		}
		std::cout << std::endl;
		std::cout.flush();
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&nw_nsets,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&nw_nbf  ,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&nw_nmo  ,2,MPI_INT,0,MPI_COMM_WORLD);

	h_basis.icgs = new double[nw_nbf];
	occ = new double[nw_nbf];
	nw_en = new double[nw_nbf];
	h_basis.nw_co = new double[nw_nbf*nw_nmo[0]];
	/*
	nw_co = new double*[nw_nbf];
	for(i=0;i<nw_nbf;i++) {
		nw_co[i] = new double[nw_nmo[0]];
	}
	*/

	if (mpi_info.sys_master) {
		if(readmode == 0) {
			//occupancy information
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\t%i\n", ignoreInt, ignoreInt); //debug
			input.read((char*) occ,ignoreInt);
//			std::cout << "occ:"; //debug
//			for(i=0;i<nw_nbf;i++) { //debug
//				std::cout << "\t" << occ[i] << std::endl; //debug
//			} //debug
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n\n", ignoreInt); //debug

			input.read((char*) &ignoreInt,4);
//			printf("%#08x\t%i\n", ignoreInt, ignoreInt); //debug
			input.read((char*) nw_en,ignoreInt);
//			std::cout << "nw_en:"; //debug
//			for(i=0;i<nw_nbf;i++) { //debug
//				std::cout << "\t" << nw_en[i] << std::endl; //debug
//			} //debug
			input.read((char*) &ignoreInt,4);
//			printf("%#08x\n\n", ignoreInt); //debug

			int index = 0;
			for(i=0;i<nw_nmo[0];i++) {
				double temp[nw_nbf];
				input.read((char*) &ignoreInt,4);
//				printf("%#08x\t%i\n", ignoreInt, ignoreInt); //debug
				input.read((char*) temp,ignoreInt);
//				std::cout << "nw_co:"; //debug
				for(j=0;j<nw_nbf;j++) {
					h_basis.nw_co[index] = temp[j];
					index++;
//					std::cout << "\t" << temp[j] << std::endl; //debug
				}
				input.read((char*) &ignoreInt,4);
//				printf("%#08x\n\n", ignoreInt); //debug
			}
		}
		else {
			for(i=0;i<nw_nbf;i++) {
				input >> occ[i];
			}
			for(i=0;i<nw_nbf;i++) {
				input >> nw_en[i];
			}

			int index = 0;
			for(i=0;i<nw_nmo[0];i++) {
				for(j=0;j<nw_nbf;j++) {
					input >> h_basis.nw_co[index];
					index++;
				}
			}
		}
		input.close();


		nw_iocc = 0;
		for(i=0;i<nw_nbf;i++) {
			if( occ[i] > 0.000 ) {
				nw_iocc = i;
			}
		}
		delete[] occ;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&nw_iocc , 1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&nw_icore, 1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(nw_en, nw_nbf, MPI_DOUBLE, 0,MPI_COMM_WORLD);
	MPI_Bcast(h_basis.nw_co, nw_nbf*nw_nmo[0], MPI_DOUBLE, 0,MPI_COMM_WORLD);

	//orbital_check();
	nw_icore = 0;
	for(i=0;i<molec.natom;i++) {
		if(molec.atom[i].znum > 3 && molec.atom[i].znum < 10)  {
			nw_icore = nw_icore + 1;
		}
	}
	iocc1 = nw_icore;
	iocc2 = nw_iocc + 1;
	ivir1 = nw_iocc + 1;
	ivir2 = nw_nmo[0];

	if (qc_ngfs != nw_nbf) {
		std::cerr << "You might use the different basis sets or geometry" << std::endl;
		exit(EXIT_FAILURE);
	}

	// print orbital energies to <JOBNAME>.orbital_energies
	if (mpi_info.sys_master) {
		std::stringstream ss;
		std::string str;
		std::ofstream output;

		// construct file name
		ss << iops.sopns[KEYS::JOBNAME] << ".orbital_energies";
		ss >> str;

		// open ofstream
		output.open(str.c_str());

		// print out data
		output << "# File contains energies for orbital " << iocc1 << " to orbital " << ivir2 << std::endl;
		output << "# 0 == first orbital" << std::endl;
		for (auto it = iocc1; it < ivir2; it++) {
			output << nw_en[it] << std::endl;
		}

		// close ofstream
		output.close();
	}
}

/*
void orbital_check(){
	int i, j;
	int znum, iocc;

	j = 0
	for(i=0;i<molec.natom;i++) {
		j = j + molec.atom[i].znum;
	}

	iocc = j/2;

	//if (iocc .ne. nw_iocc) then
	//   call qc_abort('iocc error')
	//end if


}
*/

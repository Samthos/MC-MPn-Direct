#include "qc_constant.h"

int atomic_znum(std::string param) {
	std::string atypes[] = {"H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba""Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"};
	int i = 0;
	while(param != atypes[i]) {
		i++;
	}
	if(param == atypes[i]) {
		return i+1;
	}
	return -1;
}

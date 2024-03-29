#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>

#include "../qc_mpi.h"
#include "atom_tag_parser.h"
#include "weight_function.h"

Base_Weight::Base_Weight(const MPI_info& mpi_info, const Molecule& molec, const std::string& filename) {
  read(mpi_info, molec, filename);
}
void Base_Weight::read(const MPI_info &mpi_info, const Molecule &molec, const std::string &filename) {
  int i, znum, mc_nbas;
  std::map<int, mc_basis_typ> WEIGHT_BASIS_;
  std::vector<double> alpha, coef;
  std::string atname;
  std::ifstream input;
  Atom_Tag_Parser atom_tag_parser;

  if (mpi_info.sys_master) {
    std::cout << std::endl << std::endl;
    std::cout << "Reading MC basis from " << filename << std::endl;
    input.open(filename.c_str());
    if (!input.is_open()) {
      std::cerr << filename << "does not exist" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    input >> mc_nbas >> mc_nprim;
  }

  MPI_info::barrier();
  MPI_info::broadcast_int(&mc_nbas, 1);
  MPI_info::broadcast_int(&mc_nprim, 1);

  alpha.resize(mc_nprim);
  coef.resize(mc_nprim);

  if (mpi_info.sys_master) {
    std::cout << "-------------------------------------------------------------"
                 "----------------------------------------------"
              << std::endl;
    std::cout << "\tAtom\t            contraction_exp                         coef"
              << std::endl;
    std::cout << "-------------------------------------------------------------"
                 "----------------------------------------------"
              << std::endl;
  }

  for (i = 0; i < mc_nbas; i++) {
    if (mpi_info.sys_master) {
      input >> atname;
      for (int j = 0; j < mc_nprim; j++) {  // read in primatives coefs
        input >> alpha[j] >> coef[j];
      }
      znum = atom_tag_parser.parse(atname);
      // print mc basis functions if <atom> in molecule
      if (std::any_of(molec.atoms.begin(), molec.atoms.end(),
                      [znum](Atom a) { return a.znum == znum; })) {
        std::cout << "\t " << atname << "\t";
        std::cout << std::setw(30) << std::setprecision(16) << std::fixed << alpha[0];
        std::cout << std::setw(30) << std::setprecision(16) << std::fixed << coef[0] << std::endl;
        for (int k = 1; k < mc_nprim; k++) {
          std::cout << "\t  \t";
          std::cout << std::setw(30) << std::setprecision(16) << std::fixed << alpha[k];
          std::cout << std::setw(30) << std::setprecision(16) << std::fixed << coef[k] << std::endl;
        }
        std::cout << std::endl;
      }
    }

    MPI_info::barrier();
    MPI_info::broadcast_int(&znum, 1);
    MPI_info::broadcast_double(alpha.data(), mc_nprim);
    MPI_info::broadcast_double(coef.data(), mc_nprim);

    WEIGHT_BASIS_[znum] = {alpha, coef, {0, 0, 0}};
  }

  if (mpi_info.sys_master) {
    input.close();
    std::cout << "-------------------------------------------------------------"
                 "----------------------------------------------"
              << std::endl;
    std::cout << std::setprecision(6);
    std::cout << std::endl << std::endl;
  }

  for (auto &it : molec.atoms) {
    if (WEIGHT_BASIS_.count(it.znum) == 0) {
      if (mpi_info.sys_master) {
        std::cerr << "No MC basis function defined for atom with charge "
                  << it.znum << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
  }

  for (auto &it : molec.atoms) {
    mcBasisList.push_back({WEIGHT_BASIS_[it.znum], it.pos});
  }
}

Electron_Pair_Base_Weight::Electron_Pair_Base_Weight(const MPI_info& mpi_info, const Molecule& molec, const std::string& filename) : Base_Weight(mpi_info, molec, filename) {}

Electron_Pair_GTO_Weight::Electron_Pair_GTO_Weight(const MPI_info& mpi_info, const Molecule& molec, const std::string& filename) : Electron_Pair_Base_Weight(mpi_info, molec, filename) {
  normalize();
}
double Electron_Pair_GTO_Weight::weight(const Point &pos1, const Point &pos2) const {
  double r1, r2, r12;
  double gf1, gf2;

  r12 = Point::distance(pos1, pos2);

  gf1 = 0.0;
  gf2 = 0.0;
  for (auto &it : mcBasisList) {
    r1 = Point::distance_squared(pos1, it.center);
    r2 = Point::distance_squared(pos2, it.center);

    gf1 = std::inner_product(
        it.alpha.begin(), it.alpha.end(), it.norm.begin(), gf1,
        std::plus<double>(),
        [r1](double a, double n) { return n * exp(-a * r1); });
    gf2 = std::inner_product(
        it.alpha.begin(), it.alpha.end(), it.norm.begin(), gf2,
        std::plus<double>(),
        [r2](double a, double n) { return n * exp(-a * r2); });
  }
  return gf1 * gf2 / r12;
}
void Electron_Pair_GTO_Weight::normalize() {
  int ie, je, m, ts;
  int i, j, k;
  double eri;
  double azi, azj, comz, gzi, normi, normj;
  double rr, tt, cc, h;
  double dummy;
  double g_wgt;

  // constants
  double igamma[1501][17];
  double PI = 3.141592653589793;
  double sqrt_pi = sqrt(PI);
  double pisub = 2.0 / sqrt_pi;
  double pisub2 = 2.0 * pow(sqrt_pi, 5);
  double f2[] = {-0.5, -1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5};
  double tf[] = {33.0, 37.0, 41.0, 43.0, 46.0, 49.0, 51.0, 54.0, 56.0,
                          58.0, 61.0, 63.0, 66.0, 68.0, 70.0, 72.0, 74.0};
  double f1[] = {1.0 / pisub, 1.0 / 2.0 / pisub,
                           3.0 / 4.0 / pisub, 15.0 / 8.0 / pisub,
                           105.0 / 16.0 / pisub, 945.0 / 32.0 / pisub,
                           10395.0 / 64.0 / pisub, 135135.0 / 128.0 / pisub,
                           2027025.0 / 256.0 / pisub, 34459425.0 / 512.0 / pisub };
  for (i = 0; i <= 1500; i++) {
    for (j = 0; j < 17; j++) {
      igamma[i][j] = 0.00;
    }
  }
  for (i = 0; i <= 1500; i++) {
    double a = static_cast<double>(i) / 20.0;
    double b = exp(-a);
    double c = 0.0;
    k = 50 + i / 3.0;
    for (j = k; j >= 0; j--) {
      c = (2.0 * a * c + b) / static_cast<double>(2 * j + 1);
      if (j <= 16) igamma[i][j] = c;
    }
  }

  // normalize gausians
  for (auto &it : mcBasisList) {
    std::transform(
        it.norm.begin(), it.norm.end(), it.alpha.begin(), it.norm.begin(),
        [&](double c, double a) { return c * pow(2.0 * a / PI, 0.75); });
  }

  // calculate g_wgt
  g_wgt = 0.0;
  for (auto &it : mcBasisList) {
    for (auto &jt : mcBasisList) {
      rr = Point::distance_squared(it.center, jt.center);

      for (ie = 0; ie < mc_nprim; ie++) {
        azi = it.alpha[ie];
        normi = it.norm[ie];

        for (je = 0; je < mc_nprim; je++) {
          azj = jt.alpha[je];
          normj = jt.norm[je];

          comz = azi + azj;
          gzi = azi * azj / comz;

          tt = gzi * rr;
          cc = pisub2 / (sqrt(comz) * azi * azj);

          // for only S
          m = 0;
          if (tt < tf[m]) {
            ts = round(tt * 20.0);
            h = 0.05 * static_cast<double>(ts) - tt;
            dummy =
                igamma[ts][m + 6] * h * 0.166666666666667 + igamma[ts][m + 5];
            dummy = dummy * h * 0.20 + igamma[ts][m + 4];
            dummy = dummy * h * 0.25 + igamma[ts][m + 3];
            dummy = dummy * h / 3.00 + igamma[ts][m + 2];
            dummy = dummy * h * 0.50 + igamma[ts][m + 1];
            dummy = dummy * h + igamma[ts][m + 0];
            eri = cc * dummy;
          } else {
            eri = cc * f1[m] * exp(f2[m] * log(tt));
          }
          eri = eri * normi * normj;
          cum_sum.push_back(eri);
          cum_sum_index.push_back({&it - &mcBasisList[0], &jt - &mcBasisList[0] , ie, je});
          g_wgt = g_wgt + eri;
        }
      }
    }
  }

  std::cout << "Weight normalization " << g_wgt << "\n";
  std::for_each(cum_sum.begin(), cum_sum.end(), [&](double& x) {x = x/g_wgt;});
  std::partial_sum(cum_sum.begin(), cum_sum.end(), cum_sum.begin());

  // std::cout << "g_wgt: " << g_wgt << std::endl;
  g_wgt = sqrt(g_wgt);
  for (auto &it : mcBasisList) {
    std::for_each(it.norm.begin(), it.norm.end(), [g_wgt](double& n) {n /= g_wgt;});
  }
}

Electron_Base_Weight::Electron_Base_Weight(const MPI_info& mpi_info, const Molecule& molec, const std::string& filename) : Base_Weight(mpi_info, molec, filename) {}

Electron_GTO_Weight::Electron_GTO_Weight(const MPI_info& mpi_info, const Molecule& molec, const std::string& filename) : Electron_Base_Weight(mpi_info, molec, filename) { 
  normalize();
}
double Electron_GTO_Weight::weight(const Point &pos) const {
  double weight = 0.0;
  for (auto &it : mcBasisList) {
    auto r2 = Point::distance_squared(pos, it.center);
    weight = std::inner_product(
        it.alpha.begin(), it.alpha.end(), it.norm.begin(), weight,
        std::plus<double>(),
        [r2](double a, double n) { return n * exp(-a * r2); });
  }
  return weight;
}
void Electron_GTO_Weight::normalize() {
  double norm = 0.0;
  for (auto &it : mcBasisList) {
    for (auto i = 0; i < mc_nprim; i++) {
      auto a = it.alpha[i];
      auto c = it.norm[i];
      auto n = c * 5.568327996831709 / pow(a, 1.5);
      cum_sum.push_back(n);
      cum_sum_index.push_back({&it - &mcBasisList[0], i});
      norm += n;
    }
  }

  std::for_each(cum_sum.begin(), cum_sum.end(), [&](double& x) {x = x/norm;});
  std::partial_sum(cum_sum.begin(), cum_sum.end(), cum_sum.begin());
  for (auto &it : mcBasisList) {
    std::for_each(it.norm.begin(), it.norm.end(), [norm](double& n) {n /= norm;});
  }
}

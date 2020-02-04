#include <algorithm>
#include <fstream>
#include <iomanip>


#include "../qc_mpi.h"
#include "../qc_monte.h"

GFStats::GFStats(bool isMaster_, int tasks_, int numBand, int offBand, int nDeriv, const std::string& jobname, int order) : isMaster(isMaster_), tasks(static_cast<double>(tasks_)) {
  qeps = std::vector<std::vector<double>>(numBand, std::vector<double>(nDeriv));  //stores energy correction for current step

  qepsEx1 = std::vector<std::vector<double>>(numBand, std::vector<double>(nDeriv, 0));    // stores first moment of blocked energy correction
  qepsEx2 = std::vector<std::vector<double>>(numBand, std::vector<double>(nDeriv, 0));    // stores second moment of blocked energy correction
  qepsAvg = std::vector<std::vector<double>>(numBand, std::vector<double>(nDeriv, 0));    // stores energies after MPI reduce
  qepsVar = std::vector<std::vector<double>>(numBand, std::vector<double>(nDeriv, 0));    // stores stat error after MPI reduce

  if (isMaster) {
    output_streams.resize(numBand);
    for (int band = 0; band < numBand; band++) {
      output_streams[band] = new std::ofstream[nDeriv];
      for (int deriv = 0; deriv < nDeriv; deriv++) {
        char file2[256];
        if ((band + 1 - offBand) < 0) {
          sprintf(file2, "%s.2%i.DIFF%i.HOMO%i", jobname.c_str(), order, deriv, band - offBand + 1);
        } else if ((band + 1 - offBand) == 0) {
          sprintf(file2, "%s.2%i.DIFF%i.HOMO-%i", jobname.c_str(), order, deriv, band - offBand + 1);
        } else {
          sprintf(file2, "%s.2%i.DIFF%i.LUMO+%i", jobname.c_str(), order, deriv, band - offBand);
        }
        output_streams[band][deriv].open(file2);
      }
    }
  }
}

GFStats::~GFStats() {
  if (isMaster) {
    for (uint32_t band = 0; band < qeps.size(); band++) {
      for (uint32_t deriv = 0; deriv < qeps[band].size(); deriv++) {
        output_streams[band][deriv].close();
      }
      delete[] output_streams[band];
    }
  }
}

void GFStats::blockIt(const int& step) {
  double diff1, diff2;
  uint32_t band, deriv, block;
  uint32_t blockPower2, blockStep;
  for (band = 0; band < qeps.size(); band++) {                                                         // loop over bands
    for (deriv = 0; deriv < qeps[band].size(); deriv++) {                                              // loop over number of requested derivatives
      diff1 = qeps[band][deriv] - qepsEx1[band][deriv];
      diff2 = qeps[band][deriv] * qeps[band][deriv] - qepsEx2[band][deriv];

      qepsEx1[band][deriv] += diff1 / static_cast<double>(step);
      qepsEx2[band][deriv] += diff2 / static_cast<double>(step);
    }
  }
}

void GFStats::reduce() {
  for (uint it = 0; it < qepsEx1.size(); it++) {
    MPI_info::reduce_double(qepsEx1[it].data(), qepsAvg[it].data(), qepsEx1[it].size());
    MPI_info::reduce_double(qepsEx2[it].data(), qepsVar[it].data(), qepsEx2[it].size());
  }
}

void GFStats::print(const int& step, const double& time_span) {
  double step_double = static_cast<double>(step);
  double multiplier;

  for (uint32_t band = 0; band < qepsAvg.size(); band++) {
    for (uint32_t deriv = 0; deriv < qepsAvg[band].size(); deriv++) {
      output_streams[band][deriv] << std::fixed << std::noshowpos << std::setprecision(3) << step_double / 1000.0 << "\t";  //debug2
      multiplier = 1.0 / (tasks * (static_cast<double>(step)) - 1.0);
      qepsAvg[band][deriv] = qepsAvg[band][deriv] / tasks;
      qepsVar[band][deriv] = qepsVar[band][deriv] / tasks;
      qepsVar[band][deriv] = qepsVar[band][deriv] - qepsAvg[band][deriv] * qepsAvg[band][deriv];
      qepsVar[band][deriv] = qepsVar[band][deriv] * multiplier;
      qepsVar[band][deriv] = sqrt(qepsVar[band][deriv]);

      output_streams[band][deriv] << std::fixed << std::showpos << std::setprecision(7) << qepsAvg[band][deriv] << "\t";  //debug2
      output_streams[band][deriv] << std::fixed << std::showpos << std::setprecision(7) << qepsVar[band][deriv] << "\t";  //debug2

      output_streams[band][deriv] << std::fixed << std::showpos << std::setprecision(7) << time_span << "\n";  //debug2
      output_streams[band][deriv].flush();
    }
  }
}

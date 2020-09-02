#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <vector>
#include <string>

namespace SAMPLER {
  enum SAMPLER {
    DIRECT,
    METROPOLIS,
  };
  const std::vector<std::string> sampler_strings = {
    "DIRECT",
    "METROPOLIS"
  };
}
#endif  // SAMPLER_H_

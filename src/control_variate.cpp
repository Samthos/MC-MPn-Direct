#include "control_variate.h"

Accumulator* create_accumulator(const bool& requires_blocking, const std::vector<double>& Exact_CV) {
  Accumulator* accumulator = nullptr;
  if (requires_blocking) {
    accumulator = new BlockingAccumulator(Exact_CV.size(), Exact_CV);
  } else {
    accumulator = new ControlVariate(Exact_CV.size(), Exact_CV);
  }
  return accumulator;
}

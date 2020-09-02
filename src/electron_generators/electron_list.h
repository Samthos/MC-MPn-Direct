#ifndef ELECTRON_LIST_H_
#define ELECTRON_LIST_H_

#include <array>
#include <vector>

#include "samplers.h"
#include "weight_function.h"
#include "molecule.h"
#include "../qc_random.h"

struct Electron {
  Point pos;
  double weight;
  double inverse_weight;
};
std::ostream& operator << (std::ostream& os, const Electron& electron);

class Electron_List {
 public:
  explicit Electron_List(int size);
  virtual ~Electron_List() = default;
  virtual void move(Random&, const Electron_GTO_Weight&) = 0;
  virtual bool requires_blocking() = 0;

  // functions to emulate vector interface
  std::size_t size() const {
    return electrons.size();
  }

  std::vector<Point> pos;
  std::vector<double> weight;
  std::vector<double> inverse_weight;
 protected:
  static void set_weight(Electron&, const Electron_GTO_Weight&);
  void transpose();
  std::vector<Electron>::iterator begin() {
    return electrons.begin();
  }
  std::vector<Electron>::iterator end() {
    return electrons.end();
  }

  std::vector<Electron> electrons;
};
Electron_List* create_electron_sampler(Molecule& molec,
    Electron_GTO_Weight& weight,
    int sampler_type,
    size_t electrons,
    double delx,
    int debug,
    std::string seed_file);

class Direct_Electron_List : public Electron_List {
 public:
  explicit Direct_Electron_List(int size) : Electron_List(size) {}
  ~Direct_Electron_List() override = default;
  void move(Random& random, const Electron_GTO_Weight& weight) override;
  bool requires_blocking() override;

 private:
  static void mc_move_scheme(Electron&, Random&, const Electron_GTO_Weight&);
};

class Metropolis_Electron_List : public Electron_List {
 public:
  explicit Metropolis_Electron_List(int size, double ml, Random& random, const Molecule& molec, const Electron_GTO_Weight& weight);
  ~Metropolis_Electron_List() override = default;
  void move(Random& random, const Electron_GTO_Weight& weight) override;
  bool requires_blocking() override;

 private:
  static void initialize(Electron&, Random&, const Molecule&, const Electron_GTO_Weight&);
  void mc_move_scheme(Electron&, Random&, const Electron_GTO_Weight&);
  void rescale_move_length();

  double move_length;
  int moves_since_rescale;
  int successful_moves;
  int failed_moves;
};
#endif  // ELECTRON_LIST_H_

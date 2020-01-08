#ifndef MCMP4_REF_H_
#define MCMP4_REF_H_
class MCMP4_REF {
  public:
    void mcmp4_energy(double& emp4, std::vector<double>& control4);
  private:
    void mcmp4_energy_ij(double& emp4, std::vector<double>& control);
    void mcmp4_energy_ik(double& emp4, std::vector<double>& control);
    void mcmp4_energy_il(double& emp4, std::vector<double>& control);
    void mcmp4_energy_ijkl(double& emp4, std::vector<double>& control);
};
#endif // MCMP4_REF_H_

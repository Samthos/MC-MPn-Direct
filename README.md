# MC-MPn

This project implements the Monte Carlo many-body perturbation theory (MC-MP) and Monte Carlo many-body Green's function methods (MC-GF) developed by the Hirata Lab. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In order to build MC-MPn the following are required

1. [Cmake](https://cmake.org/)
2. c++14 compliant compiler
3. A C BLAS implementation (preferable fast, i.e., [openblas](https://github.com/xianyi/OpenBLAS), mkl, etc.)
4. [Armadillo](http://arma.sourceforge.net/)
5. [LAPACK](https://performance.netlib.org/lapack/): Lapack is required for Armadillo.

For performance, it is optional but highly recommended to use an MPI compiler
for multithreading support.

### Installing

Build a simple reversion of MC-MPn is quite simple. From the top-level director run

```bash
 mkdir <build-directory>
 cd <build-directory>
 cmake .. -DCMAKE_BUILD_TYPE=Release
 make
```

If you want to build with a MPI support, substitute the cmake command with 

```bash
 MPI_CXX_COMPILER=<path-to-c++-mpi-compiler> cmake .. -DCMAKE_BUILD_TYPE=Release -DEnable_MPI=On
```

## Running MC-MPn
### Prerequisites

There are several files needed to run a calculation with MC-MPN

1. An input file (typically ending in mcin).
2. The molecular orbital coefficients and molecule orbital energies from an RHF calculation output by [**NWChem**](http://www.NWChem-sw.org/index.php/Main_Page). These are stored in the **movecs** file.
3. The molecular geometry stored in the [xyz format](https://en.wikipedia.org/wiki/XYZ_file_format). The atom specifiers in the XYZ file adopt the atom tag conventions used by [NWChem](http://www.nwchem-sw.org/index.php/Geometry#Cartesian_coordinate_input).
4. The basis set used for the RHF calculation. Basis set files can be found in \<NWChem-source-directory\>/src/basis/libraries.
5. An MC basis set to build the electron-pair importance function.

####  Notes on NWChem

The <nwchem-source-dir> should be listed in the output from the NWChem calculation.

NWChem typically rotates and translates input geometries before proceeding with any calculations.
The geometry input into MC-MPn needs to correspond to the one used to compute the MOs.
Because of this, its typically worth pulling the geometry from the NWChem output file.

The movecs file produced by NWChem is stored in a binary format.
MC-MPn can read the binary as is, although, there could potentially be issues with endianness if NWChem and MC-MPn are run on different machines.
If there are issues with the binary movecs file, convert it to an ascii format on the machine used to run NWChem.
NWChem provides a utility called mov2asc that will convert the binary into an ascii format.  See keywords for how to specify the movecs format.

####  The MC-Basis Set
  The MC-Basis is used to construct distributions for electron pairs (all methods) and for electrons (only F12 methods). The distribution for an electron pair is proportional to two sums of atom centered S-type Gaussians, one for each electron coordinate, and inversely proportional to the interelectronic distance. The distribution for an electron is proportional to a single sum  of atom centered S-type Gaussians. The sums of S-type Gaussian for the electron-pair and electron distributions are identical.

  Typically, two Gaussians are placed on each atom. If the atomic basis set contains diffuse atomic orbitals, such as the aug-cc-pVDZ basis set, using more than two Gaussians per atom may reduce the variance of the simulation.

  In the case of placing two Gaussians on each atom, reasonable values for the parameters of the Gaussians are as follows.  The height of the first Gaussian is set to be equal to the atom's number of valence electrons. Its width is approximately equal to the exponent of the most diffuse Gaussian in the most highly contracted S-type orbital of the atomic basis set.  The height of the second Gaussian is set to be one-tenth of the atom's number of valence electrons. Its width is approximately equal to the exponent of the most diffuse Gaussian in the least contracted S-type orbital of the atomic basis set.  

The format for the MC-Basis is as followed
```
<number of atoms> <guassians per atom>
<atom tag 1>
<gaussian width 1> <gaussian height 1>
. . .
<gaussian width n> <gaussian height n>
...
<atom tag m>
<gaussian width 1> <gaussian height 1>
.  .  .
<gaussian width n> <gaussian height n>
```

  See examples/input/cc-pvdz.mc_basis for an explicit example of the MC_Basis file format.

### Keywords
The input file for MC-MPn is a simple text file. Most of the options consist of keyword values pairs. 
See the examples subdirectory of the project tree for examples of valid input files.

#### High Level Options
The following are required options. All paths are relative to director where the job is launched.
* **JOBNAME**: (STRING) Name of job. Influences the names of output files.
* **GEOM**: (STRING) Path to xyz geometry file.
* **BASIS**: (STRING) Path to basis set.
* **MC_BASIS**: (STRING) Path to MC basis set.

* **MOVECS**: (STRING) Path to movecs file generated by NWChem.

The following are options technically options, but will be set for nearly every calculation.
* **JOBTYPE**: (STRING) Specifies type of calculation to perform. Default=Energy

  * ENERGY: Energy calcualtion
  * DIMER: Energy calcualtion
  * GF: Green's function calculation for diagonal element.
  * GFDIFF: Green's function calculation for diagonal element only with derivatives.
  * GFFULL: Green's function calculation for full self-energy matrix.
  * GFFULLDIFF: Green's function calculation for full self-energy matrix with derivatives.

* **TASK**: (STRING) Specifies what energy corrections to calculate for the given job type.

  * MP2: Only Energy job types. Specifies to calculate second-order correction to energy.
  * MP3: Only Energy job types. Specifies to calculate third-order correction to energy.
  * MP4: Only Energy job types. Specifies to calculate forth-order correction to energy.
  * MP2_F12_V: Only Energy job types. Calculate second-order F12V correction.
  * MP2_F12_VBX: Only Energy job types. Calculate second-order F12VBX correction.
  * GF2: Only GF job types. Specifies to calculate second-order correction to the self-energy.
  * GF3: Only GF job types. Specifies to calculate third-order correction to the self-energy.
  * GF2_F12_V: Only GF job types. Calculate second-order F12V correction. Only for occupided orbitals.
  * GF2_F12_VBX: Only GF job types. Calculate second-order F12VBX correction. Only for occupied orbitlas.

* **ELECTRON_PAIRS**: (INT) Number of electron-pair walkers to use for the calculation. Default=16
* **MC_TRIAL**: (INT) Number of MC steps to perform. Default=1024
* **FREEZE_CORE**: (INT) Option to invoke the frozen core approximation. Values={0, 1}. Default=1


#### Random Number Generation

* **DEBUG**: (INT) Values={0, 1, 2}; Default=0

  * 0: The random number generators is seeded from std::random_devices. 
  * 1: The random number generators are seeded using preset seeds. **May only run upto 16 MPI threads**
  * 2: The random number generators are seeded using the values is SEED_FILE. (See SEED_FILE options)

* **SEED_FILE**: (String); Default=""

  * DEBUG=0: The values used to seed the random number generators are saved to "SEED_FILE".
  * DEBUG=2: The values used to seed the random number generators are read from "SEED_FILE". 

* **SAMPLER**: (STRING) Sets method to use to sample the electron-pair importance function. Default=DIRECT

  * DIRECT: Use the direct sampling algorithm for the electron-pair coordinates.
  * METROPOLIS: Use the Metropolis sampling algorithm for the electron-pair coordinates. **Only if all CV_LEVELS are zero**

* **TAU_INTEGRATION**: (STRING) Sets method to integrate imaginary-time coordinates.  Default=STOCHASTIC

  * STOCHASTIC: MC integration using an [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution). Parameter is twice the HOMO LUMO gap.
  * QUADRATURE: 21-point Gauss-Kronrod quadrature rule.
  * SUPER_STOCH: MC integration using sum of exponential distributions. **Experimental**


If the sampler is set to Metropolis the following keywords may be set:
 * **MC_DELX**: (DOUBLE) Sets initial maximum offset for the Metropolis algorithm. Default=0.1
 * **NBLOCK**: (INT) May be depreciates. Set maximum number of blocking transformations. Default=1

#### MC-MP Task Options

* **MP\<N\>CV_LEVEL**: (INT) Set the deepest loop that control variates may be calculated in for energy calculations. Higher values produce more control variates. Maximum values is N.

#### MC-F12 Task Options
These options control the behavior of F12V calculations
 * **ELECTRONS**: (INT) Number of one electron walkers to use for the calculation. Default=16
 * **F12_CORRELATION_FACTOR**: (STRING) Which functional form to use for the correlation factor. A list of the available correlation factors is given below. Default=Slater.
 * **F12_GAMMA**: (DOUBLE) Value of the adjustable parameter used by the correlation factor. The default value is dependant of the correlation factor chosen.
 * **F12_BETA**: (DOUBLE) Value of the second adjustable parameter used by the correlation factor. Functionals that require a second parameter are noted. The default value is dependant of the correlation factor chosen.

MC-F12 methods have the unique ability to use nearly any function as the correlation factor, since the use of a function is not dependant of the availability of analytic integrals.  The choice of correlation factor can dramatically affect the accuracy of the resulting F12 calculation.  Because of this, it is recommend to use the Slater correlation factor as it generally performs well.  See (Cole's correlation function paper) for a detailed study comparing the usage of different correlation factors for F12 calculations.  The default values of the adjustable parameters are taken from this study.  The following correlation factors are implemented for MC-F12 calculations.
 * **Linear**: Zero-parameter functional.
 * **Rational**: One-parameter functional. Default Gamma=1.2.
 * **Slater**: One-parameter functional. Default Gamma=1.2.
 * **Slater_Linear**: One-parameter functional. Default Gamma=0.5.
 * **Gaussian**: One-parameter functional. Default Gamma=0.5.
 * **Cusped_Gaussian**: One-parameter functional. Default Gamma=1.2.
 * **Yukawa_Coulomb**: One-parameter functional. Default Gamma=2.0.
 * **Jastrow**: One-parameter functional. Default Gamma=1.2.
 * **ERFC**: One-parameter functional. Default Gamma=1.2.
 * **ERFC_Linear**: One-parameter functional. Default Gamma=0.4.
 * **Tanh**: One-parameter functional. Default Gamma=1.2.
 * **ArcTan**: One-parameter functional. Default Gamma=1.6.
 * **Logarithm**: One-parameter functional. Default Gamma=2.0.
 * **Hybrid**: One-parameter functional. Default Gamma=1.2.
 * **Two_Parameter_Rational**: Two-parameter functional. Default Gamma=NAN. Default Beta=NAN.
 * **Higher_Rational**: Two-parameter functional. Default Gamma=1.6. Default Beta=3.0.
 * **Cubic_Slater**: Two-parameter functional. Default Gamma=1.2. Default Beta=0.003.
 * **Higher_Jastrow**: Two-parameter functional. Default Gamma=0.8. Default Beta=0.75.

#### Dimer Job type Options
Dimer jobs require several additional inputs compared to a standard energy job type. The dimer job type requires three geometries and molecular orbital vectors, one each of the DIMER, MONOMER_A, and MONOMER_B subsystem. The input for the DIMER subsystem is received from the standard inputs above. The calculation assumes the all three subsystems use the same set of atomic orbitals basis.

* **MONOMER_A_GEOM**: (STRING) Path to xyz geometry file for monomer A.
* **MONOMER_B_GEOM**: (STRING) Path to xyz geometry file for monomer B.
* **MONOMER_A_MOVECS**: (STRING) Specifies monomer A molecule orbitals. Same format as MOVECS keyword.
* **MONOMER_B_MOVECS**: (STRING) Specifies monomer B molecule orbitals. Same format as MOVECS keyword.

#### GF Job type Options
These options control the sequence the behavior of MC-GF calculations.
 * **OFF_BAND**: (INT) Specifies offset of the first orbital relative to LUMO to target. Default=1 (HOMO)
 * **NUM_BAND**: (INT) Specifies number of orbitals to target. Default=1.
 * **DIFFS**: (INT) Maximum number of derivatives to calculate plus one. DIFFS=1 is a calculation with no derivatives. Diffs=2 is the self-energy  plus its first derivatives. Default=1

The OFF_BAND and NUM_BAND keywords are used in conjunction to specify the range of orbitals that the MC-GF calculation will provide a correction to.
The range of orbitals is (LUMO - OFF_BAND, ..., LUMO - OFF_BAND + NUM_BAND - 1).



### Running a Calculation
  After preparing an input files, as well as the remaining prerequisite files, launching an MP-MPn calculation may be preformed by the following

  ```
  <path-to-MP-MPn-executable> <input-file> > <output-file>
  ```

  To run a parallel MC-MPn job (assuming it was build with an MPI compiler)
  ```
  mpirun -n <number-of-threads> <path-to-MP-MPn-executable> <input-file> > <output-file>
  ```
  If the blas library used to build the executable was openblas, the launch line should be prepended with `OPENBLAS_NUM_THREADS=1`

### Output

* **STDOUT**: Basis archiving of the simulation
* **Trajectory Files**: The files archive the trajectory of the simulation. The number of steps in these files is per processor.

  The trajectory files will be named as
    * \<Jobname\>.2N for MP calculations.
    * \<Jobname\>.2N.DIFFK.Orbital for GF calculations were K specifies the derivative order.
  The N in the file name specifies which energy (self-energy) correction the trajectory follows. N=0 is the sum of all energy corrections calculation ed.
  For GF calculations, only corrections to the  diagonal elements of the self-energy are archived in a trajectory file.

  Two formats are used for these file depending on value of the SAMPLER keyword.
  * Direct: \<number-of-steps\> \<energy-estimate\> \<uncertainty-of-energy-estimate\> \<cv-energy-estimate\> \<uncertainty-cv-energy-estimate\> \<time-since-last-print\>
  * Metropolis: \<number-of-steps\> \<energy-estimate\> \<uncertainty-of-energy-estimate-blocking-transforms=0\> ... \<uncertainty-of-energy-estimate-blocking-transforms=Max\> \<time-since-last-print\>

  If the direct sampler is used in conjunction with a build that does not use control variate functions were the second (third) and fourth (fifth) columns should be identical.

* **JSONS**:
  JSON files are currently produced for the MP calculations if the simulation manages to finish. The follow similar naming conventions to the trajectory files.

* **Self-Energy Matrices**:
  TDB


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/Samthos/MC-MPn-Direct/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Alexander Doran**

See also the list of [contributors](https://github.com/Samthos/MC-MPn-Direct/contributors) who participated in this project.


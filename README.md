# MC-MPn

TBD

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

In order to build MC-MPn the following are required

1. c++14 compliant compiler
2. BLAS (preferable fast, i.e., openblas, mkl, etc.)
3. Armadillo

Optionally, but highly recommended is an MPI compiler rather than a standard c++ compiler.

### Installing

Build a simple reversion of MC-MPn is quite simple. From the top-level director run

```bash
 mkdir <build-directory>
 cd <build-directory> 
 cmake .. -DCMAKE_BUILD_TYPE=Release
 make
```

If you want to build with a MPI compiler prepend line three with

```bash
 CXX=<path-to-c++-mpi-compiler>
```

In addition to the standard cmake build options, MC-MPn has a few additional option

* -DEnable_MPI: Values={ON, OFF}; Default=Off
* -DMP2CV: Controls the number of control variates produced at the MP2 level. Values={0, 1, 2}; Default=0
* -DMP3CV: Controls the number of control variates produced at the MP3 level. Values={0, 1, 2, 3}; Default=0
* -DMP4CV: Controls the number of control variates produced at the MP4 level. Values={0, 1, 2, 3, 4}; Default=0

For the CV options, higher values correspond to more control variates being used.

## Running MC-MPn
### Prerequisites

There are several files needed to run a calculation with MC-MPN

1. An input file (typically ending in mcin).
2. The molecular orbital coefficients and molecule orbital energies from an RHF calculation output by [**NWChem**](http://www.NWChem-sw.org/index.php/Main_Page). These are stored in the **movecs** file.
3. The molecular geometry stored in the [xyz format](https://en.wikipedia.org/wiki/XYZ_file_format).
4. The basis set used for the RHF calculation, SAME FORMAT AS NWCHEM. Basis set files can be found in \<NWChem-source-directory\>/src/basis/libraries.
5. An MC basis set to build the electron-pair importance function. 

####  Notes on NWChem

The <nwchem-source-dir> should be listed in the output from the NWChem calculation.

NWChem typically rotates and translates input geometries before proceeding with any calculations. 
The geometry input into MC-MPn needs to correspond to the one used to compute the MOs. 
Because of this, its typically worth pulling the geometry from the NWChem output file.

The movecs file produced by NWChem is stored in a binary format. 
MC-MPn can read the binary as is, although, there could potentially be issues with endianness if NWChem and MC-MPn are run on different machines.
If there are issues with the binary movecs file, convert it to an ascii format on the machine used to run NWChem.
NWChem provides a utility called mov2asc that will convert the binary into an ascii format. Godspeed getting mov2asc to compile.
See keywords for how to specify the movecs format.

####  The MC-Basis Set
TBD

### Keywords
These are option to be specified in the input file. See the examples subdirectory of the project tree for examples of valid input files.

The following are required options. All paths are relative to director where the job is launched. 
* **JOBNAME**: (STRING) Name of job. Influences the names of output files.
* **GEOM**: (STRING) Path to xyz geometry file.
* **MOVECS**: (STRING) Path to movecs file generated by NWChem.
* **BASIS**: (STRING) Path to basis set.
* **MC_BASIS**: (STRING) Path to MC basis set.

The following are options technically options, but will be set for nearly every calculation.
* **TASK**: (STRING) Specifies type of calculation to perform. Default=MP

  * MP: Perturbation theory calculation.
  * GF: Green's function calculation for diagonal element.
  * GFDIFF: Green's function calculation for diagonal element only with derivatives.
  * GFFULL: Green's function calculation for full self-energy matrix.
  * GFFULLDIFF: Green's function calculation for full self-energy matrix with derivatives.

* **ORDER**: (INT) Level of theory to perform. Default=2
  
  * MP tasks are implemented through fourth order.
  * GF tasks are implemented through third order.

* **MC_NPAIR**: (INT) Number of electron-pair walkers to use for the calculation. Default=16
* **MC_TRIAL**: (INT) Number of MC steps to perform. Default=1024

The options control the sequence of random number used.

* **DEBUG**: (INT) Values={0, 1}; Default=0

  0. The random number generators is seeded from std::random_devices. 
  1. The random number generators are seeded using preset seeds. **May only run upto 16 MPI threads**

* **SAMPLER**: (STRING) Sets method to use to sample the electron-pair importance function. Default=DIRECT
  * DIRECT: Use the direct sampling algorithm for the electron-pair coordinates.
  * METROPOLIS: Use the Metropolis sampling algorithm for the electron-pair coordinates. **Not compatable with control variates**

* **TAU_INTEGRATION**: (STRING) Sets method to integrate imaginary-time coordinates.  Default=STOCHASTIC

  * STOCHASTIC: MC integration using an [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution). Parameter is twice the HOMO LUMO gap.
  * QUADRATURE: 21-point Gauss-Kronrod quadrature rule.
  * SUPER_STOCH: MC integration using sum of exponential distributions. **Experimental**


If the sampler is set to Metropolis the following keywords may be set:
 * **MC_DELX**: (DOUBLE) Sets initial maximum offset for the Metropolis algorithm. Default=0.1
 * **NBLOCK**: (INT) May be depreciates. Set maximum number of blocking transformations. Default=1

The options control the sequence the behavior of MC-GF calculations. 
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
* **Trajectory Files**: The files archive the trajectory of the simulation. The number of steps in these files is per processor and divided by 1000. 

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

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Alexander Doran**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

TBD


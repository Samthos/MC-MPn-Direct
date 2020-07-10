There are four main class provided by this library.

1. Basis_Parser
2. Basis 
3. Movec_Parser
4. Wavefunction

# Basis_Parser
The Basis_Parser class is designed to read files containing atomic orbital basis and then deliver a completed basis to the Basis class. 
It currently only read NWChem style basis set files. 
It could easily be converted to an ABC, with children reading different file formats.

It heavily makes use of
1. AtomBasis: stores the complete set of basis function for a single atom
2. Shell: stores a single (set) contracted Gaussian basis function. Auto normalizes the function on construction.

# Basis 
The basis class orchestrates the calculation of the contracted Gaussian, atomic orbital, and the AO-to-MO transformation.
It is heavily coupled with the Atomic_Orbital class which actually knows how to calculation the contraction and atomic orbital amplitudes.

The Basis class, not the Atomic_Orbital class, stores the contraction coefficient, exponents, and amplitudes. This is a deliverable choice because (to the best of my knowledge) there is no way to build a fully gpu compliant class that contains dynamic memory.

# Movec_Parser
Abstract class is designed to read the molecular orbital coefficient and molecular orbital eigenvalues in and then deliver to any class that require them, notable Wavefunction and the Tau generators.
There are currently two movec parser implements.
1. NWChem movec parser: Parser movec files produces by nwchem.
2. Dummy movec parser: Creates a set of Dummy movecs for testing purposes.

# Wavefunction
Stores the set of molecular orbitals, molecular orbital coefficient and some meta relevant meta data. Its honestly a pretty light weight class.

# TODO
## General
 - Finish gpu implementation
 - NO MORE qc_blag file names
 - Turn into a library
 - style guide
 - import optimization
 - Unify variable names
 - Less terrible names for classes like shell, atomBasis

## Basis
 - rename to atomic_basis
 - shouldn't know that wavefunction exist.
 - contraction_exp -> contraction_exponent
 - contraction_coef -> contraction_coeficien
 - ...rest of arrays


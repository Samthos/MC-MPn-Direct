#define FUNCTIONAL_NAME(name) Linear##name
#define DEFAULT_GAMMA 0.0
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Rational##name
#define DEFAULT_GAMMA 1.2
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Slater##name
#define DEFAULT_GAMMA 1.2
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Slater_Linear##name
#define DEFAULT_GAMMA 0.5
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Gaussian##name
#define DEFAULT_GAMMA 0.5
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Cusped_Gaussian##name
#define DEFAULT_GAMMA 1.2
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Yukawa_Coulomb##name
#define DEFAULT_GAMMA 2.0
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Jastrow##name
#define DEFAULT_GAMMA 1.2
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) ERFC##name
#define DEFAULT_GAMMA 1.2
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) ERFC_Linear##name
#define DEFAULT_GAMMA 0.4
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Tanh##name
#define DEFAULT_GAMMA 1.2
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) ArcTan##name
#define DEFAULT_GAMMA 1.6
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Logarithm##name
#define DEFAULT_GAMMA 2.0
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Hybrid##name
#define DEFAULT_GAMMA 1.2
#define DEFAULT_BETA  0.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Two_Parameter_Rational##name
#define DEFAULT_GAMMA 1.0
#define DEFAULT_BETA  1.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Higher_Rational##name
#define DEFAULT_GAMMA 1.6
#define DEFAULT_BETA  3.0
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Cubic_Slater##name
#define DEFAULT_GAMMA 1.2
#define DEFAULT_BETA  0.003
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

#define FUNCTIONAL_NAME(name) Higher_Jastrow##name
#define DEFAULT_GAMMA 0.8
#define DEFAULT_BETA  0.75
#include SOURCE_FILE
#undef FUNCTIONAL_NAME
#undef DEFAULT_GAMMA
#undef DEFAULT_BETA

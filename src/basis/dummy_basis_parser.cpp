#include "dummy_basis_parser.h"

Dummy_Basis_Parser::Dummy_Basis_Parser(bool is_spherical_in) {
  is_spherical = is_spherical_in;
  atomBasis[0].shell.emplace_back(SHELL::S, SHELL::Contracted_Gaussian({{33980.0000000,            {0.0000910,            -0.0000190}},
                                                                        { 5089.0000000,            {0.0007040,            -0.0001510}},
                                                                        { 1157.0000000,            {0.0036930,            -0.0007850}},
                                                                        {  326.6000000,            {0.0153600,            -0.0033240}},
                                                                        {  106.1000000,            {0.0529290,            -0.0115120}},
                                                                        {   38.1100000,            {0.1470430,            -0.0341600}},
                                                                        {   14.7500000,            {0.3056310,            -0.0771730}},
                                                                        {    6.0350000,            {0.3993450,            -0.1414930}},
                                                                        {    2.5300000,            {0.2170510,            -0.1180190}}}));
  atomBasis[0].shell.emplace_back(SHELL::S, SHELL::Contracted_Gaussian({{    0.7355000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::S, SHELL::Contracted_Gaussian({{    0.2905000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::S, SHELL::Contracted_Gaussian({{    0.1111000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::P, SHELL::Contracted_Gaussian({{   34.5100000,            {0.0053780}},
                                                                        {    7.9150000,            {0.0361320}},
                                                                        {    2.3680000,            {0.1424930}}}));
  atomBasis[0].shell.emplace_back(SHELL::P, SHELL::Contracted_Gaussian({{    0.8132000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::P, SHELL::Contracted_Gaussian({{    0.2890000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::P, SHELL::Contracted_Gaussian({{    0.1007000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::D, SHELL::Contracted_Gaussian({{    1.8480000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::D, SHELL::Contracted_Gaussian({{    0.6490000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::D, SHELL::Contracted_Gaussian({{    0.2280000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::F, SHELL::Contracted_Gaussian({{    1.4190000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::F, SHELL::Contracted_Gaussian({{    0.4850000,            {1.0000000}}}));
  atomBasis[0].shell.emplace_back(SHELL::G, SHELL::Contracted_Gaussian({{    1.0110000,            {1.0000000}}}));
}

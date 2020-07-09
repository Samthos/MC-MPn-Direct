set(source
  blas_calls.cpp
  blas_calls.h
  electron_generators/create_electron_pair_list.cpp
  electron_generators/electron_pair_list.cpp
  electron_generators/electron_pair_list.h
  electron_generators/direct_electron_pair_list.cpp
  electron_generators/direct_electron_pair_list.h
  electron_generators/metropolis_electron_pair_list.cpp
  electron_generators/metropolis_electron_pair_list.h
  electron_generators/electron_list.cpp
  electron_generators/electron_list.h
  main.cpp
  weight_function.cpp
  weight_function.h
  basis/movec_parser.h
  basis/movec_parser.cpp
  basis/create_movec_parser.cpp
  basis/nwchem_movec_parser.h
  basis/nwchem_movec_parser.cpp
  basis/dummy_movec_parser.h
  basis/dummy_movec_parser.cpp
  basis/shell.cpp
  basis/shell.h
  basis/atomic_orbital.cpp
  basis/atomic_orbital.h
  basis/basis_parser.cpp
  basis/basis_parser.h
  basis/qc_basis.cpp
  basis/qc_basis.h
  basis/qc_psi.cpp 
  basis/qc_psi_der.cpp
  basis/wavefunction.cpp 
  basis/wavefunction.h
  atom_tag_parser.h
  atom_tag_parser.cpp
  qc_geom.cpp
  qc_geom.h
  qc_input.cpp
  qc_monte.cpp
  qc_monte.h
  qc_mpi.cpp
  qc_mpi.h
  qc_ovps.h
  qc_random.cpp
  qc_random.h
  control_variate.h
  control_variate.cpp
  timer.h
  ovps_set.h
  MCGF/gf3_core_c.h
  MCGF/gf3_core_m_1.h
  MCGF/gf3_core_m_12.h
  MCGF/gf3_core_m_2.h
  MCGF/gf3_core_p_1.h
  MCGF/gf3_core_p_12.h
  MCGF/gf3_core_p_2.h
  MCGF/gf_stats.cpp
  MCGF/qc_gfFullStatisticsHost.cpp
  MCGF/qc_mcgf.cpp
  MCGF/gf_full_arrays.h
  MCGF/gf_full_arrays.cpp
  MCMP/qc_mcmp.cpp
  MCMP/mcmp.h
  MCMP/qc_mcmp2.h
  MCMP/qc_mcmp2.cpp
  MCMP/qc_mcmp3.h
  MCMP/qc_mcmp3.cpp
  MCMP/qc_mcmp4.h
  MCMP/qc_mcmp4.cpp
  MCF12/x_traces.cpp
  MCF12/x_traces.h
  MCF12/F12_Traces.cpp
  MCF12/F12_Traces.h
  MCF12/correlation_factors.cpp
  MCF12/correlation_factors.h
  MCF12/mp2_f12.cpp
  MCF12/mp2_f12.h
  MCF12/gf2_f12.cpp
  MCF12/gf2_f12.h
  MCGF/qc_gfFullStatistics.cpp
  MCGF/qc_mcgf.h
  MCGF/qc_mcgf2.h
  MCGF/qc_mcgf3.h
  MCGF/qc_mcgf2.cpp
  MCGF/qc_mcgf3.cpp 
  cpu_qc_monte.cpp
  MCMP/qc_mcmp4_ijk.h MCMP/qc_mcmp4_ijkl.h
  MCMP/qc_mcmp4_ij.h MCMP/qc_mcmp4_ij_k.h MCMP/qc_mcmp4_ij_l.h
  MCMP/qc_mcmp4_ik.h MCMP/qc_mcmp4_ik_j.h MCMP/qc_mcmp4_ik_l.h
  MCMP/qc_mcmp4_il.h MCMP/qc_mcmp4_il_j.h MCMP/qc_mcmp4_il_k.h
)

add_subdirectory(tau_generators)

set(CPU_SPECIFIC_SOURCES
  ovps_set.cpp
  qc_ovps.cpp 
)

add_executable(MC_MPn_Direct ${source} ${CPU_SPECIFIC_SOURCES})

target_link_libraries(MC_MPn_Direct ${ARMADILLO_LIBRARIES})
target_link_libraries(MC_MPn_Direct ${BLAS_LIBRARIES})
target_link_libraries(MC_MPn_Direct tau)

target_include_directories(MC_MPn_Direct PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_include_directories(MC_MPn_Direct PUBLIC ${BLAS_INCLUDE_DIRS})
target_include_directories(MC_MPn_Direct PUBLIC ./tau_generators)
target_include_directories(MC_MPn_Direct PUBLIC ./electron_generators)

IF(${MPI_CXX_FOUND})
  message("Adding MPI compile options to MC_MPn_Direct")
  target_compile_definitions(MC_MPn_Direct PUBLIC -DHAVE_MPI)
  target_link_libraries(MC_MPn_Direct ${MPI_CXX_LIBRARIES})
  target_include_directories(MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()

if(${CUDA_FOUND})
  set(GPU_SPECIFIC_SOURCES
    ovps_set.cu
    qc_ovps.cu
  )

  cuda_add_executable(GPU_MC_MPn_Direct ${source} ${GPU_SPECIFIC_SOURCES})
  target_link_libraries(GPU_MC_MPn_Direct ${ARMADILLO_LIBRARIES})
  target_link_libraries(GPU_MC_MPn_Direct ${BLAS_LIBRARIES})
  target_link_libraries(GPU_MC_MPn_Direct tau)

  target_include_directories(GPU_MC_MPn_Direct PUBLIC ${ARMADILLO_INCLUDE_DIRS})
  target_include_directories(GPU_MC_MPn_Direct PUBLIC ${BLAS_INCLUDE_DIRS})
  target_include_directories(GPU_MC_MPn_Direct PUBLIC ./tau_generators)
  target_include_directories(GPU_MC_MPn_Direct PUBLIC ./electron_generators)

  cuda_add_cublas_to_target(GPU_MC_MPn_Direct)
  target_compile_definitions(GPU_MC_MPn_Direct PUBLIC -DHAVE_CUDA)
  IF(${MPI_CXX_FOUND})
    message("Adding MPI compile options to GPU_MC_MPn_Direct")
    target_compile_definitions(GPU_MC_MPn_Direct PUBLIC -DHAVE_MPI)
    target_link_libraries(GPU_MC_MPn_Direct ${MPI_CXX_LIBRARIES})
    target_include_directories(GPU_MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
  endif()
endif()


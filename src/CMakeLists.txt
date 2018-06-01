include_directories(${ARMADILLO_INCLUDE_DIRS})

set(source
  blas_calls.cpp
  blas_calls.h
  cpu_qc_monte.cpp
#  cublasStatus_t_getErrorString.cu
#  cublasStatus_t_getErrorString.h
  el_pair.cpp
  el_pair.h
  gf2_core.h
  gf3_core_c.h
  gf3_core_m_1.h
  gf3_core_m_12.h
  gf3_core_m_2.h
  gf3_core_p_1.h
  gf3_core_p_12.h
  gf3_core_p_2.h
  gf_stats.cpp
#  gpu_gfFullStatistics.cu
#  gpu_mcgf2.cu
#  gpu_mcgf3.cu
#  gpu_ovps.cu
#  gpu_psi.cu
#  gpu_qc_monte.cpp
  main.cpp
  mc_basis.cpp
  mc_basis.h
  nw_vectors.cpp
  qc_basis.cpp
  qc_basis.h
  qc_constant.cpp
  qc_constant.h
  qc_geom.cpp
  qc_geom.h
  qc_gfFullStatistics.cpp
  qc_gfFullStatisticsHost.cpp
  qc_input.cpp
  qc_input.h
  qc_mcgf2.cpp
  qc_mcgf3.cpp
  qc_monte.cpp
  qc_monte.h
  qc_mpi.cpp
  qc_mpi.h
  qc_ovps.cpp
  qc_ovps.h
  qc_psi.cpp
  qc_random.cpp
  qc_random.h
)

add_executable(MC_MPn_Direct ${source})
target_link_libraries(MC_MPn_Direct ${ARMADILLO_LIBRARIES})
IF(${MPI_CXX_FOUND})
    message("Adding MPI compile options to MC_MPn_Direct")
    target_compile_definitions(MC_MPn_Direct PUBLIC -DUSE_MPI)
    target_link_libraries(MC_MPn_Direct ${MPI_CXX_LIBRARIES})
    target_include_directories(MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()


#add_executable(MC_MPn_Direct_Stoch ${source})
#target_link_libraries(MC_MPn_Direct_Stoch ${ARMADILLO_LIBRARIES})
#target_compile_definitions(MC_MPn_Direct_Stoch PUBLIC -DUSE_MC_T)
#IF(${MPI_CXX_FOUND})
#    target_compile_definitions(MC_MPn_Direct_Stoch PUBLIC -DUSE_MPI)
#    target_link_libraries(MC_MPn_Direct_Stoch ${MPI_CXX_LIBRARIES})
#    target_include_directories(MC_MPn_Direct_Stoch PRIVATE ${MPI_CXX_INCLUDE_PATH})
#endif()

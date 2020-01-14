set(source
        blas_calls.cpp
        blas_calls.h
        electron_pair_list.cpp
        electron_pair_list.h
        electron_list.cpp
        electron_list.h
        main.cpp
        weight_function.cpp
        weight_function.h
        basis/nw_vectors.cpp
        basis/qc_basis.cpp
        basis/qc_basis.h
        atom_znum.cpp
        atom_znum.h
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
        timer.h
        tau_integrals.h
        tau_integrals.cpp
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
        MCMP/qc_mcmp.cpp
        MCMP/qc_mcmp2.cpp
        MCMP/qc_mcmp3.cpp
        MCMP/qc_mcmp4.cpp
        MCF12/F12_Traces.cpp
        MCF12/F12_Traces.h
        MCF12/correlation_factors.cpp
        MCF12/correlation_factors.h
        MCF12/mp2_f12_integrals.cpp
        MCF12/mp2f12_var.h
)

set(CPU_EXTRA_SOURCES
        MCGF/qc_gfFullStatistics.cpp MCGF/qc_mcgf2.cpp MCGF/qc_mcgf3.cpp qc_ovps.cpp basis/qc_psi.cpp cpu_qc_monte.cpp
        MCMP/qc_mcmp4_ijk.h MCMP/qc_mcmp4_ijkl.h
        MCMP/qc_mcmp4_ij.h MCMP/qc_mcmp4_ij_k.h MCMP/qc_mcmp4_ij_l.h
        MCMP/qc_mcmp4_ik.h MCMP/qc_mcmp4_ik_j.h MCMP/qc_mcmp4_ik_l.h
        MCMP/qc_mcmp4_il.h MCMP/qc_mcmp4_il_j.h MCMP/qc_mcmp4_il_k.h
        )


# include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(MC_MPn_Direct ${source} ${CPU_EXTRA_SOURCES})

target_link_libraries(MC_MPn_Direct ${ARMADILLO_LIBRARIES})
target_link_libraries(MC_MPn_Direct ${BLAS_LIBRARIES})

target_include_directories(MC_MPn_Direct PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_include_directories(MC_MPn_Direct PUBLIC ${BLAS_INCLUDE_DIRS})

target_compile_definitions(MC_MPn_Direct PUBLIC -DMP2CV=${MP2CV})
target_compile_definitions(MC_MPn_Direct PUBLIC -DMP3CV=${MP3CV})
target_compile_definitions(MC_MPn_Direct PUBLIC -DMP4CV=${MP4CV})

IF(${MPI_CXX_FOUND})
    message("Adding MPI compile options to MC_MPn_Direct")
    target_compile_definitions(MC_MPn_Direct PUBLIC -DHAVE_MPI)
    target_link_libraries(MC_MPn_Direct ${MPI_CXX_LIBRARIES})
    target_include_directories(MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()

#set(GPU_EXTRA_SOURCES
#    cublasStatus_t_getErrorString.cu cublasStatus_t_getErrorString.h
#    MCGF/gpu_gfFullStatistics.cu MCGF/gpu_mcgf2.cu MCGF/gpu_mcgf3.cu gpu_ovps.cu basis/gpu_psi.cu gpu_qc_monte.cpp)
#cuda_add_executable(GPU_MC_MPn_Direct ${source} ${GPU_EXTRA_SOURCES})
#target_link_libraries(GPU_MC_MPn_Direct ${ARMADILLO_LIBRARIES})
#target_compile_definitions(GPU_MC_MPn_Direct PUBLIC -DHAVE_CUDA)
#cuda_add_cublas_to_target(GPU_MC_MPn_Direct)
#IF(${MPI_CXX_FOUND})
#    message("Adding MPI compile options to GPU_MC_MPn_Direct")
#    target_compile_definitions(GPU_MC_MPn_Direct PUBLIC -DHAVE_MPI)
#    target_link_libraries(GPU_MC_MPn_Direct ${MPI_CXX_LIBRARIES})
#    target_include_directories(GPU_MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
#endif()


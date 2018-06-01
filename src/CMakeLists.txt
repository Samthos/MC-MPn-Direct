include_directories(${ARMADILLO_INCLUDE_DIRS})

set(source
        gf2_core.h gf3_core_c.h gf3_core_m_1.h gf3_core_m_12.h gf3_core_m_2.h gf3_core_p_1.h gf3_core_p_12.h gf3_core_p_2.h
        blas_calls.cpp
        blas_calls.h
        el_pair.cpp
        el_pair.h
        gf_stats.cpp
        main.cpp
        weight_function.cpp
        weight_function.h
        nw_vectors.cpp
        qc_basis.cpp
        qc_basis.h
        qc_constant.cpp
        qc_constant.h
        qc_geom.cpp
        qc_geom.h
        qc_gfFullStatisticsHost.cpp
        qc_input.cpp
        qc_monte.cpp
        qc_monte.h
        qc_mpi.cpp
        qc_mpi.h
        qc_ovps.h
        qc_random.cpp
        qc_random.h
)

set(CPU_EXTRA_SOURCES
        qc_gfFullStatistics.cpp qc_mcgf2.cpp qc_mcgf3.cpp qc_ovps.cpp qc_psi.cpp cpu_qc_monte.cpp
)

set(GPU_EXTRA_SOURCES
        cublasStatus_t_getErrorString.cu cublasStatus_t_getErrorString.h
        gpu_gfFullStatistics.cu gpu_mcgf2.cu gpu_mcgf3.cu gpu_ovps.cu gpu_psi.cu gpu_qc_monte.cpp
)
add_executable(MC_MPn_Direct ${source} ${CPU_EXTRA_SOURCES})
target_link_libraries(MC_MPn_Direct ${ARMADILLO_LIBRARIES})
IF(${MPI_CXX_FOUND})
    message("Adding MPI compile options to MC_MPn_Direct")
    target_compile_definitions(MC_MPn_Direct PUBLIC -DUSE_MPI)
    target_link_libraries(MC_MPn_Direct ${MPI_CXX_LIBRARIES})
    target_include_directories(MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()


add_executable(GPU_MC_MPn_Direct ${source} ${GPU_EXTRA_SOURCES})
target_link_libraries(GPU_MC_MPn_Direct ${ARMADILLO_LIBRARIES})
target_link_libraries(GPU_MC_MPn_Direct cublas)
IF(${MPI_CXX_FOUND})
    message("Adding MPI compile options to GPU_MC_MPn_Direct")
    target_compile_definitions(GPU_MC_MPn_Direct PUBLIC -DUSE_MPI)
    target_link_libraries(GPU_MC_MPn_Direct ${MPI_CXX_LIBRARIES})
    target_include_directories(GPU_MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()


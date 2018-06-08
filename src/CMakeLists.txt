include_directories(${ARMADILLO_INCLUDE_DIRS})

set(source
        MCGF/gf2_core.h MCGF/gf3_core_c.h MCGF/gf3_core_m_1.h MCGF/gf3_core_m_12.h MCGF/gf3_core_m_2.h MCGF/gf3_core_p_1.h MCGF/gf3_core_p_12.h MCGF/gf3_core_p_2.h
        MCGF/gf_stats.cpp
        MCGF/qc_gfFullStatisticsHost.cpp
        MCGF/qc_mcgf.cpp
        blas_calls.cpp
        blas_calls.h
        el_pair.cpp
        el_pair.h
        main.cpp
        weight_function.cpp
        weight_function.h
        basis/nw_vectors.cpp
        basis/qc_basis.cpp
        basis/qc_basis.h
        qc_constant.cpp
        qc_constant.h
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
        MCMP/qc_mcmp.cpp
        MCMP/qc_mcmp2.cpp
        MCMP/qc_mcmp3.cpp
        tau_integrals.h
)

set(CPU_EXTRA_SOURCES
        MCGF/qc_gfFullStatistics.cpp MCGF/qc_mcgf2.cpp MCGF/qc_mcgf3.cpp qc_ovps.cpp basis/qc_psi.cpp cpu_qc_monte.cpp
        )
set(GPU_EXTRA_SOURCES
        cublasStatus_t_getErrorString.cu cublasStatus_t_getErrorString.h
        MCGF/gpu_gfFullStatistics.cu MCGF/gpu_mcgf2.cu MCGF/gpu_mcgf3.cu gpu_ovps.cu basis/gpu_psi.cu gpu_qc_monte.cpp)


add_executable(MC_MPn_Direct ${source} ${CPU_EXTRA_SOURCES})
target_link_libraries(MC_MPn_Direct ${ARMADILLO_LIBRARIES})
IF(${MPI_CXX_FOUND})
    message("Adding MPI compile options to MC_MPn_Direct")
    target_compile_definitions(MC_MPn_Direct PUBLIC -DHAVE_MPI)
    target_link_libraries(MC_MPn_Direct ${MPI_CXX_LIBRARIES})
    target_include_directories(MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()



add_executable(GPU_MC_MPn_Direct ${source} ${GPU_EXTRA_SOURCES})
target_link_libraries(GPU_MC_MPn_Direct ${ARMADILLO_LIBRARIES})

target_compile_definitions(GPU_MC_MPn_Direct PUBLIC -DHAVE_CUDA)
target_link_libraries(GPU_MC_MPn_Direct cublas)
target_include_directories(GPU_MC_MPn_Direct PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

IF(${MPI_CXX_FOUND})
    message("Adding MPI compile options to GPU_MC_MPn_Direct")
    target_compile_definitions(GPU_MC_MPn_Direct PUBLIC -DHAVE_MPI)
    target_link_libraries(GPU_MC_MPn_Direct ${MPI_CXX_LIBRARIES})
    target_include_directories(GPU_MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()


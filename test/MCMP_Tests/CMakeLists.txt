message("MCMP2 test " ${gtest_SOURCE_DIR}/)

include_directories(${gtest_SOURCE_DIR}/include)
include_directories(${gtest_SOURCE_DIR})
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(MCMP_Tests MCMP_Tests.cpp ../ovps_test/ovps_Tests.cpp)
target_link_libraries(MCMP_Tests ${ARMADILLO_LIBRARIES})
target_link_libraries(MCMP_Tests gtest_main)
target_link_libraries(MCMP_Tests gtest)

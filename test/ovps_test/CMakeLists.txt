message("ovps test " ${gtest_SOURCE_DIR}/)

include_directories(${gtest_SOURCE_DIR}/include)
include_directories(${gtest_SOURCE_DIR})
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(ovps_Tests ovps_Tests.cpp)
target_link_libraries(ovps_Tests ${ARMADILLO_LIBRARIES})
target_link_libraries(ovps_Tests gtest_main)
target_link_libraries(ovps_Tests gtest)

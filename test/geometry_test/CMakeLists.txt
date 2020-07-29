add_executable(point_test.x
  point_test.cpp
)
target_include_directories(point_test.x PUBLIC ../../src/geometry)
target_link_libraries(point_test.x gtest_main)
target_link_libraries(point_test.x geometry)
add_test(NAME POINT_TEST COMMAND point_test.x)

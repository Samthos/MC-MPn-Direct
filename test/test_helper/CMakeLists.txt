set(test_helper_source
  test_helper.cpp
)

add_library(test_helper STATIC ${test_helper_source})
if(${Enable_CUDA})
  set_source_files_properties(${test_helper_source} PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(test_helper PRIVATE -DHAVE_CUDA)
endif()


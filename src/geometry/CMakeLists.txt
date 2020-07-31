set(source
  point.h
  atom.h
  atom.cpp
  atom_tag_parser.cpp
  atom_tag_parser.h
  molecule.cpp
  molecule.h
)

if(${Enable_CUDA})
list(APPEND source point.cu)
else()
list(APPEND source point.cpp)
endif()

add_library(geometry STATIC ${source})
if(${Enable_CUDA})
  set_target_properties(geometry PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(geometry PRIVATE -DHAVE_CUDA)
endif()

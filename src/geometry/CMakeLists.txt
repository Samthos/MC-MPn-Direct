set(source
  atom.h
  atom.cpp
  atom_tag_parser.cpp
  atom_tag_parser.h
  molecule.cpp
  molecule.h
)

add_library(geometry STATIC ${source})

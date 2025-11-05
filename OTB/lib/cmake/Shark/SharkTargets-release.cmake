#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "shark" for configuration "Release"
set_property(TARGET shark APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(shark PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/shark.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/shark.dll"
  )

list(APPEND _cmake_import_check_targets shark )
list(APPEND _cmake_import_check_files_for_shark "${_IMPORT_PREFIX}/lib/shark.lib" "${_IMPORT_PREFIX}/bin/shark.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

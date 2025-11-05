#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GSL::gsl" for configuration "Release"
set_property(TARGET GSL::gsl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GSL::gsl PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/gsl.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/gsl.dll"
  )

list(APPEND _cmake_import_check_targets GSL::gsl )
list(APPEND _cmake_import_check_files_for_GSL::gsl "${_IMPORT_PREFIX}/lib/gsl.lib" "${_IMPORT_PREFIX}/bin/gsl.dll" )

# Import target "GSL::gslcblas" for configuration "Release"
set_property(TARGET GSL::gslcblas APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GSL::gslcblas PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/gslcblas.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/gslcblas.dll"
  )

list(APPEND _cmake_import_check_targets GSL::gslcblas )
list(APPEND _cmake_import_check_files_for_GSL::gslcblas "${_IMPORT_PREFIX}/lib/gslcblas.lib" "${_IMPORT_PREFIX}/bin/gslcblas.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

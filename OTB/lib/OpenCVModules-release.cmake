#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "opencv_core" for configuration "Release"
set_property(TARGET opencv_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_core PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_core460.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_core460.dll"
  )

list(APPEND _cmake_import_check_targets opencv_core )
list(APPEND _cmake_import_check_files_for_opencv_core "${_IMPORT_PREFIX}/lib/opencv_core460.lib" "${_IMPORT_PREFIX}/bin/opencv_core460.dll" )

# Import target "opencv_ml" for configuration "Release"
set_property(TARGET opencv_ml APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_ml PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_ml460.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_ml460.dll"
  )

list(APPEND _cmake_import_check_targets opencv_ml )
list(APPEND _cmake_import_check_files_for_opencv_ml "${_IMPORT_PREFIX}/lib/opencv_ml460.lib" "${_IMPORT_PREFIX}/bin/opencv_ml460.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

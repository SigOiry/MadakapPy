#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "xdr-static" for configuration "Release"
set_property(TARGET xdr-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(xdr-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libxdr.lib"
  )

list(APPEND _cmake_import_check_targets xdr-static )
list(APPEND _cmake_import_check_files_for_xdr-static "${_IMPORT_PREFIX}/lib/libxdr.lib" )

# Import target "xdr-shared" for configuration "Release"
set_property(TARGET xdr-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(xdr-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/xdr.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/xdr.dll"
  )

list(APPEND _cmake_import_check_targets xdr-shared )
list(APPEND _cmake_import_check_files_for_xdr-shared "${_IMPORT_PREFIX}/lib/xdr.lib" "${_IMPORT_PREFIX}/bin/xdr.dll" )

# Import target "dfalt-static" for configuration "Release"
set_property(TARGET dfalt-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(dfalt-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libdfalt.lib"
  )

list(APPEND _cmake_import_check_targets dfalt-static )
list(APPEND _cmake_import_check_files_for_dfalt-static "${_IMPORT_PREFIX}/lib/libdfalt.lib" )

# Import target "dfalt-shared" for configuration "Release"
set_property(TARGET dfalt-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(dfalt-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/dfalt.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/dfalt.dll"
  )

list(APPEND _cmake_import_check_targets dfalt-shared )
list(APPEND _cmake_import_check_files_for_dfalt-shared "${_IMPORT_PREFIX}/lib/dfalt.lib" "${_IMPORT_PREFIX}/bin/dfalt.dll" )

# Import target "mfhdfalt-static" for configuration "Release"
set_property(TARGET mfhdfalt-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mfhdfalt-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmfhdfalt.lib"
  )

list(APPEND _cmake_import_check_targets mfhdfalt-static )
list(APPEND _cmake_import_check_files_for_mfhdfalt-static "${_IMPORT_PREFIX}/lib/libmfhdfalt.lib" )

# Import target "mfhdfalt-shared" for configuration "Release"
set_property(TARGET mfhdfalt-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mfhdfalt-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/mfhdfalt.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/mfhdfalt.dll"
  )

list(APPEND _cmake_import_check_targets mfhdfalt-shared )
list(APPEND _cmake_import_check_files_for_mfhdfalt-shared "${_IMPORT_PREFIX}/lib/mfhdfalt.lib" "${_IMPORT_PREFIX}/bin/mfhdfalt.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

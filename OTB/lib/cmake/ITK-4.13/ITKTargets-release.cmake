#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "itkdouble-conversion" for configuration "Release"
set_property(TARGET itkdouble-conversion APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(itkdouble-conversion PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/itkdouble-conversion-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/itkdouble-conversion-4.13.dll"
  )

list(APPEND _cmake_import_check_targets itkdouble-conversion )
list(APPEND _cmake_import_check_files_for_itkdouble-conversion "${_IMPORT_PREFIX}/lib/itkdouble-conversion-4.13.lib" "${_IMPORT_PREFIX}/bin/itkdouble-conversion-4.13.dll" )

# Import target "itksys" for configuration "Release"
set_property(TARGET itksys APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(itksys PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/itksys-4.13.lib"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "ws2_32;Psapi"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/itksys-4.13.dll"
  )

list(APPEND _cmake_import_check_targets itksys )
list(APPEND _cmake_import_check_files_for_itksys "${_IMPORT_PREFIX}/lib/itksys-4.13.lib" "${_IMPORT_PREFIX}/bin/itksys-4.13.dll" )

# Import target "itkvcl" for configuration "Release"
set_property(TARGET itkvcl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(itkvcl PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/itkvcl-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/itkvcl-4.13.dll"
  )

list(APPEND _cmake_import_check_targets itkvcl )
list(APPEND _cmake_import_check_files_for_itkvcl "${_IMPORT_PREFIX}/lib/itkvcl-4.13.lib" "${_IMPORT_PREFIX}/bin/itkvcl-4.13.dll" )

# Import target "itknetlib" for configuration "Release"
set_property(TARGET itknetlib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(itknetlib PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/itknetlib-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/itknetlib-4.13.dll"
  )

list(APPEND _cmake_import_check_targets itknetlib )
list(APPEND _cmake_import_check_files_for_itknetlib "${_IMPORT_PREFIX}/lib/itknetlib-4.13.lib" "${_IMPORT_PREFIX}/bin/itknetlib-4.13.dll" )

# Import target "itkv3p_netlib" for configuration "Release"
set_property(TARGET itkv3p_netlib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(itkv3p_netlib PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/itkv3p_netlib-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/itkv3p_netlib-4.13.dll"
  )

list(APPEND _cmake_import_check_targets itkv3p_netlib )
list(APPEND _cmake_import_check_files_for_itkv3p_netlib "${_IMPORT_PREFIX}/lib/itkv3p_netlib-4.13.lib" "${_IMPORT_PREFIX}/bin/itkv3p_netlib-4.13.dll" )

# Import target "itkvnl" for configuration "Release"
set_property(TARGET itkvnl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(itkvnl PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/itkvnl-4.13.lib"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "itkvcl"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/itkvnl-4.13.dll"
  )

list(APPEND _cmake_import_check_targets itkvnl )
list(APPEND _cmake_import_check_files_for_itkvnl "${_IMPORT_PREFIX}/lib/itkvnl-4.13.lib" "${_IMPORT_PREFIX}/bin/itkvnl-4.13.dll" )

# Import target "itkvnl_algo" for configuration "Release"
set_property(TARGET itkvnl_algo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(itkvnl_algo PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/itkvnl_algo-4.13.lib"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "itknetlib;itkv3p_netlib;itkvnl"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/itkvnl_algo-4.13.dll"
  )

list(APPEND _cmake_import_check_targets itkvnl_algo )
list(APPEND _cmake_import_check_files_for_itkvnl_algo "${_IMPORT_PREFIX}/lib/itkvnl_algo-4.13.lib" "${_IMPORT_PREFIX}/bin/itkvnl_algo-4.13.dll" )

# Import target "ITKVNLInstantiation" for configuration "Release"
set_property(TARGET ITKVNLInstantiation APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKVNLInstantiation PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKVNLInstantiation-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKVNLInstantiation-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKVNLInstantiation )
list(APPEND _cmake_import_check_files_for_ITKVNLInstantiation "${_IMPORT_PREFIX}/lib/ITKVNLInstantiation-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKVNLInstantiation-4.13.dll" )

# Import target "ITKCommon" for configuration "Release"
set_property(TARGET ITKCommon APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKCommon PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKCommon-4.13.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "itkdouble-conversion"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKCommon-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKCommon )
list(APPEND _cmake_import_check_files_for_ITKCommon "${_IMPORT_PREFIX}/lib/ITKCommon-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKCommon-4.13.dll" )

# Import target "itkNetlibSlatec" for configuration "Release"
set_property(TARGET itkNetlibSlatec APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(itkNetlibSlatec PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/itkNetlibSlatec-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/itkNetlibSlatec-4.13.dll"
  )

list(APPEND _cmake_import_check_targets itkNetlibSlatec )
list(APPEND _cmake_import_check_files_for_itkNetlibSlatec "${_IMPORT_PREFIX}/lib/itkNetlibSlatec-4.13.lib" "${_IMPORT_PREFIX}/bin/itkNetlibSlatec-4.13.dll" )

# Import target "ITKStatistics" for configuration "Release"
set_property(TARGET ITKStatistics APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKStatistics PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKStatistics-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKStatistics-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKStatistics )
list(APPEND _cmake_import_check_files_for_ITKStatistics "${_IMPORT_PREFIX}/lib/ITKStatistics-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKStatistics-4.13.dll" )

# Import target "ITKTransform" for configuration "Release"
set_property(TARGET ITKTransform APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKTransform PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKTransform-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKTransform-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKTransform )
list(APPEND _cmake_import_check_files_for_ITKTransform "${_IMPORT_PREFIX}/lib/ITKTransform-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKTransform-4.13.dll" )

# Import target "ITKLabelMap" for configuration "Release"
set_property(TARGET ITKLabelMap APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKLabelMap PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKLabelMap-4.13.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "ITKStatistics"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKLabelMap-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKLabelMap )
list(APPEND _cmake_import_check_files_for_ITKLabelMap "${_IMPORT_PREFIX}/lib/ITKLabelMap-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKLabelMap-4.13.dll" )

# Import target "ITKMesh" for configuration "Release"
set_property(TARGET ITKMesh APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKMesh PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKMesh-4.13.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "ITKTransform"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKMesh-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKMesh )
list(APPEND _cmake_import_check_files_for_ITKMesh "${_IMPORT_PREFIX}/lib/ITKMesh-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKMesh-4.13.dll" )

# Import target "ITKMetaIO" for configuration "Release"
set_property(TARGET ITKMetaIO APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKMetaIO PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKMetaIO-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKMetaIO-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKMetaIO )
list(APPEND _cmake_import_check_files_for_ITKMetaIO "${_IMPORT_PREFIX}/lib/ITKMetaIO-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKMetaIO-4.13.dll" )

# Import target "ITKSpatialObjects" for configuration "Release"
set_property(TARGET ITKSpatialObjects APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKSpatialObjects PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKSpatialObjects-4.13.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "ITKCommon;ITKMesh"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKSpatialObjects-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKSpatialObjects )
list(APPEND _cmake_import_check_files_for_ITKSpatialObjects "${_IMPORT_PREFIX}/lib/ITKSpatialObjects-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKSpatialObjects-4.13.dll" )

# Import target "ITKPath" for configuration "Release"
set_property(TARGET ITKPath APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKPath PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKPath-4.13.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "ITKCommon"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKPath-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKPath )
list(APPEND _cmake_import_check_files_for_ITKPath "${_IMPORT_PREFIX}/lib/ITKPath-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKPath-4.13.dll" )

# Import target "ITKQuadEdgeMesh" for configuration "Release"
set_property(TARGET ITKQuadEdgeMesh APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKQuadEdgeMesh PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKQuadEdgeMesh-4.13.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "ITKMesh"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKQuadEdgeMesh-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKQuadEdgeMesh )
list(APPEND _cmake_import_check_files_for_ITKQuadEdgeMesh "${_IMPORT_PREFIX}/lib/ITKQuadEdgeMesh-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKQuadEdgeMesh-4.13.dll" )

# Import target "ITKIOImageBase" for configuration "Release"
set_property(TARGET ITKIOImageBase APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKIOImageBase PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKIOImageBase-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKIOImageBase-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKIOImageBase )
list(APPEND _cmake_import_check_files_for_ITKIOImageBase "${_IMPORT_PREFIX}/lib/ITKIOImageBase-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKIOImageBase-4.13.dll" )

# Import target "ITKOptimizers" for configuration "Release"
set_property(TARGET ITKOptimizers APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKOptimizers PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKOptimizers-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKOptimizers-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKOptimizers )
list(APPEND _cmake_import_check_files_for_ITKOptimizers "${_IMPORT_PREFIX}/lib/ITKOptimizers-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKOptimizers-4.13.dll" )

# Import target "ITKPolynomials" for configuration "Release"
set_property(TARGET ITKPolynomials APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKPolynomials PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKPolynomials-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKPolynomials-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKPolynomials )
list(APPEND _cmake_import_check_files_for_ITKPolynomials "${_IMPORT_PREFIX}/lib/ITKPolynomials-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKPolynomials-4.13.dll" )

# Import target "ITKBiasCorrection" for configuration "Release"
set_property(TARGET ITKBiasCorrection APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKBiasCorrection PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKBiasCorrection-4.13.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "ITKCommon"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKBiasCorrection-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKBiasCorrection )
list(APPEND _cmake_import_check_files_for_ITKBiasCorrection "${_IMPORT_PREFIX}/lib/ITKBiasCorrection-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKBiasCorrection-4.13.dll" )

# Import target "ITKFFT" for configuration "Release"
set_property(TARGET ITKFFT APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKFFT PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKFFT-4.13.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "ITKCommon"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKFFT-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKFFT )
list(APPEND _cmake_import_check_files_for_ITKFFT "${_IMPORT_PREFIX}/lib/ITKFFT-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKFFT-4.13.dll" )

# Import target "ITKTransformFactory" for configuration "Release"
set_property(TARGET ITKTransformFactory APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKTransformFactory PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKTransformFactory-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKTransformFactory-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKTransformFactory )
list(APPEND _cmake_import_check_files_for_ITKTransformFactory "${_IMPORT_PREFIX}/lib/ITKTransformFactory-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKTransformFactory-4.13.dll" )

# Import target "ITKIOTransformBase" for configuration "Release"
set_property(TARGET ITKIOTransformBase APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKIOTransformBase PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKIOTransformBase-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKIOTransformBase-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKIOTransformBase )
list(APPEND _cmake_import_check_files_for_ITKIOTransformBase "${_IMPORT_PREFIX}/lib/ITKIOTransformBase-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKIOTransformBase-4.13.dll" )

# Import target "ITKIOTransformInsightLegacy" for configuration "Release"
set_property(TARGET ITKIOTransformInsightLegacy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKIOTransformInsightLegacy PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKIOTransformInsightLegacy-4.13.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "itkdouble-conversion"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKIOTransformInsightLegacy-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKIOTransformInsightLegacy )
list(APPEND _cmake_import_check_files_for_ITKIOTransformInsightLegacy "${_IMPORT_PREFIX}/lib/ITKIOTransformInsightLegacy-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKIOTransformInsightLegacy-4.13.dll" )

# Import target "ITKIOTransformMatlab" for configuration "Release"
set_property(TARGET ITKIOTransformMatlab APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKIOTransformMatlab PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKIOTransformMatlab-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKIOTransformMatlab-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKIOTransformMatlab )
list(APPEND _cmake_import_check_files_for_ITKIOTransformMatlab "${_IMPORT_PREFIX}/lib/ITKIOTransformMatlab-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKIOTransformMatlab-4.13.dll" )

# Import target "ITKKLMRegionGrowing" for configuration "Release"
set_property(TARGET ITKKLMRegionGrowing APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKKLMRegionGrowing PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKKLMRegionGrowing-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKKLMRegionGrowing-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKKLMRegionGrowing )
list(APPEND _cmake_import_check_files_for_ITKKLMRegionGrowing "${_IMPORT_PREFIX}/lib/ITKKLMRegionGrowing-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKKLMRegionGrowing-4.13.dll" )

# Import target "itklbfgs" for configuration "Release"
set_property(TARGET itklbfgs APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(itklbfgs PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/itklbfgs-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/itklbfgs-4.13.dll"
  )

list(APPEND _cmake_import_check_targets itklbfgs )
list(APPEND _cmake_import_check_files_for_itklbfgs "${_IMPORT_PREFIX}/lib/itklbfgs-4.13.lib" "${_IMPORT_PREFIX}/bin/itklbfgs-4.13.dll" )

# Import target "ITKOptimizersv4" for configuration "Release"
set_property(TARGET ITKOptimizersv4 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKOptimizersv4 PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKOptimizersv4-4.13.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "itklbfgs"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKOptimizersv4-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKOptimizersv4 )
list(APPEND _cmake_import_check_files_for_ITKOptimizersv4 "${_IMPORT_PREFIX}/lib/ITKOptimizersv4-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKOptimizersv4-4.13.dll" )

# Import target "ITKWatersheds" for configuration "Release"
set_property(TARGET ITKWatersheds APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ITKWatersheds PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ITKWatersheds-4.13.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ITKWatersheds-4.13.dll"
  )

list(APPEND _cmake_import_check_targets ITKWatersheds )
list(APPEND _cmake_import_check_files_for_ITKWatersheds "${_IMPORT_PREFIX}/lib/ITKWatersheds-4.13.lib" "${_IMPORT_PREFIX}/bin/ITKWatersheds-4.13.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

#-----------------------------------------------------------------------------
#
# ITKConfig.cmake - ITK CMake configuration file for external projects.
#
# This file is configured by ITK and used by the UseITK.cmake module
# to load ITK's settings for an external project.
#
# ITK components can be specified when using the CMake function `find_package()`
# E.g. find_package(ITK COMPONENTS ITKCommon ITKIOImageBase ITKIONRRD ITKImageGradient)
#
# The components listed in this call must be ITK module names or factories
# (e.g. ITKImageIO, ITKTransformIO). Factories allow the user to load all the
# modules of a specific type (e.g. ImageIO, TransformIO) without having to
# specify them individually.
# See `itk_module_config()` documentation for more information about factory modules.


# Compute the installation prefix from this ITKConfig.cmake file location.
get_filename_component(ITK_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(ITK_INSTALL_PREFIX "${ITK_INSTALL_PREFIX}" PATH)
get_filename_component(ITK_INSTALL_PREFIX "${ITK_INSTALL_PREFIX}" PATH)
get_filename_component(ITK_INSTALL_PREFIX "${ITK_INSTALL_PREFIX}" PATH)
set(ITK_MODULES_DIR "${ITK_INSTALL_PREFIX}/lib/cmake/ITK-4.13/Modules")

# The C and C++ flags added by ITK to the cmake-configured flags.
set(ITK_REQUIRED_C_FLAGS "")
set(ITK_REQUIRED_CXX_FLAGS " /bigobj")
set(ITK_REQUIRED_LINK_FLAGS "")

# The ITK version number
set(ITK_VERSION_MAJOR "4")
set(ITK_VERSION_MINOR "13")
set(ITK_VERSION_PATCH "3")

# If ITK was built with version 3 compatibility features.
set(ITKV3_COMPATIBILITY "OFF")

# Remove all legacy code completely.
set(ITK_LEGACY_REMOVE "OFF")

# Silence all legacy code messages.
set(ITK_LEGACY_SILENT "OFF")

# Remove code that will become legacy in future releases completely.
set(ITK_FUTURE_LEGACY_REMOVE "OFF")

# ITK's CMake directory with modules ITK uses.
set(ITK_CMAKE_DIR "${ITK_INSTALL_PREFIX}/lib/cmake/ITK-4.13")

# The location of the UseITK.cmake file.
set(ITK_USE_FILE "${ITK_INSTALL_PREFIX}/lib/cmake/ITK-4.13/UseITK.cmake")

# Whether ITK was built with shared libraries.
set(ITK_BUILD_SHARED "ON")

# Whether ITK examples were built.
set(ITK_BUILD_EXAMPLES "OFF")

# Whether ITK documentation was built.
set(ITK_BUILD_DOCUMENTATION "OFF")

# List of available ITK modules.
set(ITK_MODULES_ENABLED "ITKDoubleConversion;ITKKWIML;ITKKWSys;ITKVNL;ITKVNLInstantiation;ITKCommon;ITKFiniteDifference;ITKImageFilterBase;ITKCurvatureFlow;ITKImageAdaptors;ITKNetlib;ITKStatistics;ITKTransform;ITKImageFunction;ITKImageGrid;ITKAnisotropicSmoothing;ITKLabelMap;ITKImageCompose;ITKMesh;ITKZLIB;ITKMetaIO;ITKSpatialObjects;ITKImageStatistics;ITKPath;ITKImageIntensity;ITKThresholding;ITKConnectedComponents;ITKMathematicalMorphology;ITKBinaryMathematicalMorphology;ITKImageLabel;ITKNarrowBand;ITKDistanceMap;ITKQuadEdgeMesh;ITKFastMarching;ITKIOImageBase;ITKImageCompare;ITKSmoothing;ITKImageGradient;ITKImageSources;ITKImageFeature;ITKOptimizers;ITKSignedDistanceFunction;ITKLevelSets;ITKAntiAlias;ITKPolynomials;ITKBiasCorrection;ITKClassifiers;ITKColormap;ITKFFT;ITKConvolution;ITKDeconvolution;ITKDeformableMesh;ITKDenoising;ITKDisplacementField;ITKEigen;ITKGPUCommon;ITKGPUFiniteDifference;ITKGPUAnisotropicSmoothing;ITKGPUImageFilterBase;ITKGPURegistrationCommon;ITKRegistrationCommon;ITKPDEDeformableRegistration;ITKGPUPDEDeformableRegistration;ITKGPUSmoothing;ITKGPUThresholding;ITKTransformFactory;ITKIOTransformBase;ITKIOTransformInsightLegacy;ITKIOTransformMatlab;ITKImageFusion;ITKKLMRegionGrowing;ITKLIBLBFGS;ITKLabelVoting;ITKLevelSetsv4;ITKMarkovRandomFieldsClassifiers;ITKOptimizersv4;ITKMetricsv4;ITKNeuralNetworks;ITKQuadEdgeMeshFiltering;ITKRegionGrowing;ITKRegistrationMethodsv4;ITKSpatialFunction;ITKVoronoi;ITKWatersheds")

# Import ITK targets.
set(ITK_CONFIG_TARGETS_FILE "${ITK_INSTALL_PREFIX}/lib/cmake/ITK-4.13/ITKTargets.cmake")
if(NOT ITK_TARGETS_IMPORTED)
  set(ITK_TARGETS_IMPORTED 1)
  include("${ITK_CONFIG_TARGETS_FILE}")
endif()

# Load module interface macros.
include("${ITK_INSTALL_PREFIX}/lib/cmake/ITK-4.13/ITKModuleAPI.cmake")
# Load requested modules and their dependencies into variables:
#  ITK_LIBRARIES       = Libraries to link
#  ITK_INCLUDE_DIRS    = Header file search path
#  ITK_LIBRARY_DIRS    = Library search path (for outside dependencies)
#  ITK_RUNTIME_LIBRARY_DIRS = Runtime linker search path
#  ITK_FACTORY_NAMES = List of <module>::<factory>::<format>
#  ITK_FACTORY_LIST  = List of factories set
#  ITK_<factory_name> = List of formats for each factory
itk_module_config(ITK ${ITK_MODULES_ENABLED})

# Compute set of requested modules.
if(ITK_FIND_COMPONENTS)
  set(ITK_MODULES_REQUESTED ${ITK_FIND_COMPONENTS})
  # Specific modules requested by find_package(ITK).
  itk_module_config(ITK ${ITK_MODULES_REQUESTED})
else()
  set(ITK_MODULES_REQUESTED ${ITK_MODULES_ENABLED})
endif()

# Add configuration with FFTW
set(ITK_USE_FFTWD "ON")
set(ITK_USE_FFTWF "ON")
set(ITK_USE_SYSTEM_FFTW "ON")
set(ITK_FFTW_INCLUDE_PATH "C:/build/otb/xdk/include")
set(ITK_FFTW_LIBDIR "C:/build/otb/xdk/lib")

# Add FFTW include and library directories
if (ITK_USE_FFTWF OR ITK_USE_FFTWD)
  set(ITK_INCLUDE_DIRS ${ITK_INCLUDE_DIRS} "${ITK_FFTW_INCLUDE_PATH}")
  set(ITK_LIBRARY_DIRS ${ITK_LIBRARY_DIRS} "${ITK_FFTW_LIBDIR}")
endif()

# Add configuration with GPU
set(ITK_USE_GPU "OFF")

# Wrapping
set(ITK_WRAPPING "OFF")
if( NOT DEFINED ITK_WRAP_PYTHON)
  set(ITK_WRAP_PYTHON "OFF")
endif()
if( NOT DEFINED ITK_WRAP_JAVA)
  set(ITK_WRAP_JAVA "OFF")
endif()
if(NOT DEFINED ITK_WRAP_RUBY)
  set(ITK_WRAP_RUBY "OFF")
endif()
if( NOT DEFINED ITK_WRAP_PERL)
  set(ITK_WRAP_PERL "OFF")
endif()
if(NOT DEFINED ITK_WRAP_TCL)
  set(ITK_WRAP_TCL "OFF")
endif()

set(ITK_WRAP_PYTHON_VERSION "3.7.9")

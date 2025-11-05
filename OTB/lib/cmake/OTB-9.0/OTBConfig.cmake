#
# Copyright (C) 2005-2022 Centre National d'Etudes Spatiales (CNES)
#
# This file is part of Orfeo Toolbox
#
#     https://www.orfeo-toolbox.org/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#-----------------------------------------------------------------------------
#
# OTBConfig.cmake - OTB CMake configuration file for external projects.
#
# This file is configured by OTB and used by the UseOTB.cmake module
# to load OTB's settings for an external project.

# Compute the installation prefix from this OTBConfig.cmake file location.
get_filename_component(OTB_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(OTB_INSTALL_PREFIX "${OTB_INSTALL_PREFIX}" PATH)
get_filename_component(OTB_INSTALL_PREFIX "${OTB_INSTALL_PREFIX}" PATH)
get_filename_component(OTB_INSTALL_PREFIX "${OTB_INSTALL_PREFIX}" PATH)
set(OTB_MODULES_DIR "${OTB_INSTALL_PREFIX}/lib/cmake/OTB-9.0/Modules")
set(OTB_PYTHONPATH "${OTB_INSTALL_PREFIX}/lib/otb/python")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#RK: TODO: add sse2, see4, see4.1, openmp flags

# The C and C++ flags added by OTB to the cmake-configured flags.
set(OTB_REQUIRED_C_FLAGS "")
set(OTB_REQUIRED_CXX_FLAGS " /bigobj")
set(OTB_REQUIRED_LINK_FLAGS " /MANIFEST:NO")

# The OTB version number
set(OTB_VERSION_MAJOR "9")
set(OTB_VERSION_MINOR "0")
set(OTB_VERSION_PATCH "0")

# OTB's CMake directory with modules OTB uses.
set(OTB_CMAKE_DIR "${OTB_INSTALL_PREFIX}/lib/cmake/OTB-9.0")

# OTB uses OPENMP
set(OTB_USE_OPENMP "OFF")

# OTB uses SSE optim
set(OTB_USE_SSE_FLAGS "ON")

# use Find*.cmake files included in OTB package for external dependencies. (OpenCV, GDAL etc..)
set(CMAKE_MODULE_PATH ${OTB_CMAKE_DIR} ${CMAKE_MODULE_PATH})

# The location of the UseOTB.cmake file.
set(OTB_USE_FILE "${OTB_INSTALL_PREFIX}/lib/cmake/OTB-9.0/UseOTB.cmake")

# Whether OTB was built with shared libraries.
set(OTB_BUILD_SHARED "ON")

# Default value for WINDOWS_EXPORT_ALL_SYMBOLS target property
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS "ON")

# List of available OTB modules.
set(OTB_MODULES_ENABLED "OTB6S;OTBBoost;OTBBoostAdapters;OTBGDAL;OTBITK;OTBCommon;OTBDate;OTBGdalAdapters;OTBObjectList;OTBTinyXML;OTBMetadata;OTBImageBase;OTBFunctor;OTBAnomalyDetection;OTBVectorDataBase;OTBIOGDAL;OTBExtendedFilename;OTBImageList;OTBStreaming;OTBImageIO;OTBInterpolation;OTBTransform;OTBImageManipulation;OTBTestKernel;OTBPath;OTBVectorDataManipulation;OTBProjection;OTBVectorDataIO;OTBApplicationEngine;OTBCurl;OTBCurlAdapters;OTBCarto;OTBMoments;OTBLabelMap;OTBConversion;OTBIOXML;OTBOpticalCalibration;OTBConvolution;OTBMosaic;OTBStatistics;OTBPanSharpening;OTBAppCore;OTBCommandLine;OTBChangeDetection;OTBColorMap;OTBContrast;OTBPointSet;OTBSiftFast;OTBDescriptors;OTBDimensionalityReduction;OTBEdge;OTBFuzzy;OTBIndices;OTBMuParser;OTBMathParser;OTBMuParserX;OTBMathParserX;OTBSmoothing;OTBTextures;OTBAppFeaturesExtraction;OTBEndmembersExtraction;OTBWavelet;OTBUnmixing;OTBAppHyperspectral;OTBShark;OTBLearningBase;OTBUnsupervised;OTBLibSVM;OTBOpenCV;OTBSupervised;OTBDempsterShafer;OTBMajorityVoting;OTBSOM;OTBSampling;OTBAppLearning;OTBDimensionalityReductionLearning;OTBMorphologicalProfiles;OTBAppMiscellaneous;OTBImageNoise;OTBPolarimetry;OTBSARCalibration;OTBAppSAR;OTBCCOBIA;OTBMeanShift;OTBMetrics;OTBLabelling;OTBOGRProcessing;OTBWatersheds;OTBAppSegmentation;OTBDEM;OTBStereo;OTBDisparityMap;OTBAppStereo;OTBCloudDetection;OTBComplexImage;OTBCorner;OTBDensity;OTBGSL;OTBLandSatClassifier;OTBMarkov;OTBQGIS;OTBSWIGWrapper;OTBSimulation;OTBTemporalGapFilling;OTBTimeSeries;S1TilingSupportApplications;SertitObject;otbGRM")

set(OTB_APPLICATION_PATH "${OTB_INSTALL_PREFIX}/lib/otb/applications")

# Import OTB targets.
set(OTB_CONFIG_TARGETS_FILE "${OTB_INSTALL_PREFIX}/lib/cmake/OTB-9.0/OTBTargets.cmake")
if(NOT OTB_TARGETS_IMPORTED)
  set(OTB_TARGETS_IMPORTED 1)
  include("${OTB_CONFIG_TARGETS_FILE}")
endif()

# Load module interface macros.
include("${OTB_INSTALL_PREFIX}/lib/cmake/OTB-9.0/OTBModuleAPI.cmake")

# Load application macros.
include(CMakeParseArguments)
include("${OTB_INSTALL_PREFIX}/lib/cmake/OTB-9.0/OTBApplicationMacros.cmake")

# Compute set of requested modules.
if(OTB_FIND_COMPONENTS)
  # Specific modules requested by find_package(OTB).
  set(OTB_MODULES_REQUESTED "${OTB_FIND_COMPONENTS}")
endif()
if(NOT OTB_MODULES_REQUESTED)
  # No specific modules requested.  Use all of them.
  set(OTB_MODULES_REQUESTED "${OTB_MODULES_ENABLED}")
endif()

# Load requested modules and their dependencies into variables:
#  OTB_LIBRARIES       = Libraries to link
#  OTB_INCLUDE_DIRS    = Header file search path
#  OTB_LIBRARY_DIRS    = Library search path (for outside dependencies)
otb_module_config(OTB ${OTB_MODULES_REQUESTED})

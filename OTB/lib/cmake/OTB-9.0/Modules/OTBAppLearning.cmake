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

set(OTBAppLearning_LOADED 1)
set(OTBAppLearning_DEPENDS "OTBApplicationEngine;OTBBoost;OTBColorMap;OTBCommon;OTBConversion;OTBDempsterShafer;OTBFunctor;OTBGdalAdapters;OTBIOXML;OTBITK;OTBImageBase;OTBImageManipulation;OTBIndices;OTBInterpolation;OTBLearningBase;OTBMajorityVoting;OTBMathParser;OTBObjectList;OTBProjection;OTBSOM;OTBSampling;OTBStatistics;OTBStreaming;OTBSupervised;OTBUnsupervised;OTBVectorDataBase;OTBVectorDataIO;OTBVectorDataManipulation")
set(OTBAppLearning_LIBRARIES "OTBApplicationEngine;Boost::filesystem;Boost::serialization;Boost::unit_test_framework;ITKCommon;itksys;itkvnl_algo;itkvnl;itkv3p_netlib;itknetlib;itkvcl;ITKStatistics;ITKTransform;itkNetlibSlatec;ITKMesh;ITKQuadEdgeMesh;ITKSpatialObjects;ITKMetaIO;C:/build/otb/xdk/lib/zdll.lib;ITKLabelMap;ITKPath;ITKIOImageBase;ITKOptimizers;ITKBiasCorrection;ITKPolynomials;ITKFFT;ITKOptimizersv4;ITKKLMRegionGrowing;ITKWatersheds;OTBCommon;C:/build/otb/xdk/lib/gdal.lib;OTBGdalAdapters;OTBIOGDAL;OTBImageBase;OTBVectorDataBase;OTBImageManipulation;OTBTransform;OTBMetadata;OTBStreaming;OTBFuzzy;OTBMathParser;OTBSupervised;C:/build/otb/xdk/lib/tinyxml.lib;OTBLearningBase;OTBProjection;OTBSampling;OTBStatistics;C:/build/otb/xdk/lib/shark.lib;OTBVectorDataIO")
set(OTBAppLearning_INCLUDE_DIRS "${OTB_INSTALL_PREFIX}/include/OTB-9.0")
set(OTBAppLearning_LIBRARY_DIRS "")


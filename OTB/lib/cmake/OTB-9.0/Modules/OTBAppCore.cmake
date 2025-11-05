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

set(OTBAppCore_LOADED 1)
set(OTBAppCore_DEPENDS "OTBApplicationEngine;OTBBoostAdapters;OTBCarto;OTBCommon;OTBConversion;OTBCurlAdapters;OTBFunctor;OTBIOXML;OTBITK;OTBImageBase;OTBImageManipulation;OTBObjectList;OTBOpticalCalibration;OTBPanSharpening;OTBProjection;OTBStatistics;OTBStreaming;OTBTransform;OTBVectorDataBase;OTBVectorDataManipulation")
set(OTBAppCore_LIBRARIES "OTBApplicationEngine;Boost::filesystem;Boost::serialization;Boost::unit_test_framework;OTBCarto;OTBCommon;C:/build/otb/xdk/lib/gdal.lib;OTBGdalAdapters;OTBIOGDAL;ITKCommon;itksys;itkvnl_algo;itkvnl;itkv3p_netlib;itknetlib;itkvcl;ITKStatistics;ITKTransform;itkNetlibSlatec;ITKMesh;ITKQuadEdgeMesh;ITKSpatialObjects;ITKMetaIO;C:/build/otb/xdk/lib/zdll.lib;ITKLabelMap;ITKPath;ITKIOImageBase;ITKOptimizers;ITKBiasCorrection;ITKPolynomials;ITKFFT;ITKOptimizersv4;ITKKLMRegionGrowing;ITKWatersheds;OTBImageBase;OTBVectorDataBase;OTBImageManipulation;OTBTransform;OTBMetadata;OTBStreaming;OTBCurlAdapters;C:/build/otb/xdk/lib/tinyxml.lib;OTBOpticalCalibration;OTBStatistics;OTBProjection")
set(OTBAppCore_INCLUDE_DIRS "${OTB_INSTALL_PREFIX}/include/OTB-9.0")
set(OTBAppCore_LIBRARY_DIRS "")


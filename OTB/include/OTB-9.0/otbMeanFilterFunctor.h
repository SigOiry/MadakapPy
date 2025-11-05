/*
 * Copyright (C) 2005-2020 Centre National d'Etudes Spatiales (CNES)
 *
 * This file is part of S1Tiling remote module for Orfeo Toolbox
 *
 *     https://www.orfeo-toolbox.org/
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef otbMeanFilterFunctor_h
#define otbMeanFilterFunctor_h

#include "itkNumericTraits.h"

namespace otb
{

namespace Functor
{

/**
 * - compute the ratio of the two pixel values
 * - compute the value of the ratio of means
 * - cast the \c double value resulting to the pixel type of the output image
 * - store the casted value into the output image.
 *
 * \ingroup Functor
 * \ingroup OTBChangeDetection
 * \ingroup S1TilingSupportApplications
 */
template<class TInput1, class TOutput>
class MeanFilterFunctor
{
  using PixelNumericType = typename itk::NumericTraits<TOutput>::ValueType;
public:
  MeanFilterFunctor() = default;
  TOutput operator ()(const TInput1& itA) const
  {
    TOutput meanA=static_cast<TOutput>(0.0);
    meanA[0]=0.;

    int non_zeros_pixels = 0;
    for (unsigned long pos = 0; pos < itA.Size(); ++pos)
    {
      if (itA.GetPixel(pos)[0]!=0.)
      {
        meanA[0] += itA.GetPixel(pos)[0];
        ++non_zeros_pixels;
      }
    }

    if (non_zeros_pixels > 0)
    {
      meanA[0] /= static_cast<PixelNumericType>(non_zeros_pixels);
    }
    else
    {
      meanA[0] = 0.;
    }

    //std::cout<<"meanA= "<<meanA<<", meanB= "<<meanB<<std::endl;

    TOutput ratio=itA.GetCenterPixel();
    if (ratio[0]!=0.)
    {
      ratio[0]=meanA[0];
    }
    //std::cout<<ratio[0]<<"\n";
    return static_cast<TOutput>(ratio);
  }
};
}
} // end namespace otb

#endif

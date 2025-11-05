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

#ifndef otbOutcoreFunctor_h
#define otbOutcoreFunctor_h

namespace otb
{

namespace Functor
{

/**
 * Computes the outcore for the multitemp filter.
 *
 * \ingroup Functor
 * \ingroup OTBMultiTempFilter
 * \ingroup S1TilingSupportApplications
 */
template<class TInput, class TOutput>
class OutcoreFunctor
{
public:
  OutcoreFunctor() = default;
  ~OutcoreFunctor() = default;
  TOutput operator ()(const TInput& itA) const
  {
    TOutput meanA;
    meanA.SetSize(1);
    meanA.Fill(0.);
    TOutput pixel;
    pixel.SetSize(1);

    TOutput nb_non_zero_pixels;
    nb_non_zero_pixels.SetSize(1);
    nb_non_zero_pixels.Fill(0.);

    for (unsigned long pos = 0; pos < itA.Size(); ++pos)
    {
      pixel = itA.GetPixel(pos);
      meanA += pixel;
      for (unsigned long i = 0; i < pixel.Size(); ++i)
      {
        if (pixel[i] != 0.)
        {
          nb_non_zero_pixels[i]++;
        }
      }
    }
    for (unsigned long i = 0; i < pixel.Size(); ++i)
    {
      if (nb_non_zero_pixels[i] !=0.)
      {
        meanA[i] /= nb_non_zero_pixels[i];
      }
      else
      {
        meanA[i]=0.;
      }
    }

    TOutput ratio=itA.GetCenterPixel();
    ratio.SetSize(meanA.GetSize()+1);   // add an element to store the ENL (nb of images used for temporal average)

    for (unsigned long i = 0; i < meanA.Size(); ++i)
    {
      if ((meanA[i]!=0.)&&(ratio[i]!=0.))
      {
        ratio[i]/=meanA[i];
      }
      else
      {
        ratio[i]=0.;
      }
      // Compute the ENL (nb of images used for temporal average)
      ratio[meanA.Size()]=int(ratio[0]>0.);
    }
    return static_cast<TOutput>(ratio);
  }
};
}
} // end namespace otb

#endif

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

#ifndef otbMultiplyVectorImageFilter_h
#define otbMultiplyVectorImageFilter_h
#include "otbMultiplyVectorImageFilterFunctor.h"
#include "itkBinaryFunctorImageFilter.h"

namespace otb
{

/** \class MeanRatioImageFilter
 * \brief Implements neighborhood-wise the computation of mean ratio.
 *
 * This filter is parametrized over the types of the two
 * input images and the type of the output image.
 *
 * Numeric conversions (castings) are done by the C++ defaults.
 *
 * The filter will walk over all the pixels in the two input images, and for
 * each one of them it will do the following:
 *
 * - compute the ratio of the two pixel values
 * - compute the value of the ratio of means
 * - cast the \c double value resulting to the pixel type of the output image
 * - store the casted value into the output image.
 *
 * The filter expect all images to have the same dimension
 * (e.g. all 2D, or all 3D, or all ND)
 *
 * \ingroup IntensityImageFilters Multithreaded
 * \ingroup OTBChangeDetection
 * \ingroup S1TilingSupportApplications
 */

template <class TInputImage1, class TInputImage2, class TOutputImage>
class ITK_EXPORT MultiplyVectorImageFilter :
  public itk::BinaryFunctorImageFilter<TInputImage1, TInputImage2, TOutputImage,
      Functor::MultiplyVectorImageFilterFunctor<typename TInputImage1::PixelType,typename TInputImage2::PixelType,typename TOutputImage::PixelType> >
{
public:
  /**\name Standard class typedefs */
  //@{
  using Self         = MultiplyVectorImageFilter;
  using Superclass   = itk::BinaryFunctorImageFilter<
      TInputImage1, TInputImage1,TOutputImage,
      Functor::MultiplyVectorImageFilterFunctor<typename TInputImage1::PixelType,typename TInputImage2::PixelType,typename TOutputImage::PixelType>
      >;
  using Pointer      = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  //@}

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Macro defining the type*/
  itkTypeMacro(MultiplyVectorImageFilter, SuperClass);

protected:
  MultiplyVectorImageFilter() = default;
  ~MultiplyVectorImageFilter() override = default;

private:
  MultiplyVectorImageFilter(const Self &) = delete;
  Self& operator =(const Self&) = delete;
};

} // end namespace otb

#endif

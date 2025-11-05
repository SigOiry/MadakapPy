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

#ifndef otbMultiplyVectorImageFilterFunctor_h
#define otbMultiplyVectorImageFilterFunctor_h

namespace otb
{
namespace Functor
{

template<class TInput1, class TInput2, class TOutputPixel>
class MultiplyVectorImageFilterFunctor
{
public:
  TOutputPixel operator ()(const TInput1& p1, const TInput2& p2) const
  {
    unsigned int const nbComp = p1.GetSize();
    TOutputPixel outValue(nbComp);
    for(unsigned int i = 0; i<nbComp;++i)
    {
      outValue[i] = static_cast<typename TOutputPixel::ValueType>(p1[i]*p2[i]/p2[1]);
    }
    return outValue;
  }
};
}
} // end namespace otb

#endif

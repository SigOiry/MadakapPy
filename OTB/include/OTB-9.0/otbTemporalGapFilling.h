/*=========================================================================

  Program:   gapfilling
  Language:  C++

  Copyright (c) Jordi Inglada. All rights reserved.

  See LICENSE for details.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _TEMPORALGAPFILLING_H_
#define _TEMPORALGAPFILLING_H_

#include "otbVectorImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"
#include "otbStandardFilterWatcher.h"
#include "itkBinaryFunctorImageFilter.h"
#include <vector>
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include "otbDateUtils.h"

namespace GapFilling
{

template<typename ValueType, typename VectorType>
inline
itk::VariableLengthVector<ValueType> vectorToPixel(const VectorType& v)
{
  itk::VariableLengthVector<ValueType> p(v.size());
  for(auto i=0; i<p.Size(); ++i)
    p[i] = ValueType{v[i]};
  return p;
}

template<typename ValueType>
inline
itk::VariableLengthVector<ValueType> vectorToPixel(const std::vector<ValueType>& v)
{
  itk::VariableLengthVector<ValueType> p(v.size());
  for(size_t i=0; i<p.Size(); ++i)
    p[i] = ValueType{v[i]};
  return p;
}
/** Binary functor image filter which produces a vector image with a
* number of bands different from the input images */
template <class TInputImage1, class TInputImage2, class TOutputImage, 
          class TFunctor>
class ITK_EXPORT BinaryFunctorImageFilterWithNBands : 
    public itk::BinaryFunctorImageFilter< TInputImage1, TInputImage2, 
                                          TOutputImage, TFunctor >
{
public:
  typedef BinaryFunctorImageFilterWithNBands Self;
  typedef itk::BinaryFunctorImageFilter< TInputImage1, TInputImage2, 
                                         TOutputImage, TFunctor > Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Macro defining the type*/
  itkTypeMacro(BinaryFunctorImageFilterWithNBands, SuperClass);

  /** Accessors for the number of bands*/
  itkSetMacro(NumberOfOutputBands, unsigned int);
  itkGetConstMacro(NumberOfOutputBands, unsigned int);
  
protected:
  BinaryFunctorImageFilterWithNBands() = default;
  virtual ~BinaryFunctorImageFilterWithNBands() = default;

  void GenerateOutputInformation() override
  {
    Superclass::GenerateOutputInformation();
    this->GetOutput()->SetNumberOfComponentsPerPixel( m_NumberOfOutputBands );
  }
private:
  BinaryFunctorImageFilterWithNBands(const Self &) = delete; //purposely not implemented
  void operator =(const Self&) = delete; //purposely not implemented

  unsigned int m_NumberOfOutputBands;


};
/** Return 2 vectors containing, for each date, the position of the
* last (resp next) valid date and a bool which is true if there are no
* valid dates */
template <typename PixelType>
inline
std::tuple<std::vector<typename PixelType::ValueType>,
           std::vector<typename PixelType::ValueType>, bool>
find_valid_bounds(const PixelType mask, int nbDates,
                  typename PixelType::ValueType valid_value)
{
  using PValueType = typename PixelType::ValueType;
  using PVectorType = typename std::vector<PValueType>;

  PVectorType l_valid(nbDates,0);
  PVectorType n_valid(nbDates,0);
  int lv{-1};
  int nv{nbDates};
  for(auto i=0; i<nbDates; i++)
    {
    if(mask[i]==(valid_value)) lv=i;
    l_valid[i] = lv;
    auto j = nbDates-1-i;
    if(mask[j]==(valid_value)) nv=j;
    n_valid[j] = nv;
    }
  return std::make_tuple(l_valid, n_valid, (lv==-1 && nv==nbDates));
}
/**  Generate a new pixel and a corresponding mask after interlacing
*  input and output dates. The resulting pixel values for the dates
*  which were not available in the input have a nundefined value and
*  are masked.
*/
template <typename PixelType>
inline
std::tuple<PixelType, PixelType, PixelType >
create_tmp_data_interlace_dates(const PixelType pix, 
                                const PixelType mask, 
                                const PixelType idv, 
                                const PixelType odv,
                                typename PixelType::ValueType valid_value)
{
  if(idv == odv)
    return std::make_tuple(pix, mask, odv);

  unsigned int nbDates = idv.GetSize() + odv.GetSize();
  PixelType opix{nbDates};
  PixelType omask{nbDates};
  PixelType dv{nbDates};

  unsigned int dcount = 0;
  unsigned int icount = 0;
  unsigned int ocount = 0;

  while(dcount < nbDates)
    {
    if(icount < idv.GetSize() &&
       (ocount == odv.GetSize() || //ouput dates consumed
        idv[icount] <= odv[ocount]))
      {
      opix[dcount] = pix[icount];
      omask[dcount] = mask[icount];
      dv[dcount] = idv[icount];
      icount++;
      }
    else
      {
      opix[dcount] = typename PixelType::ValueType{0};
      omask[dcount] = valid_value+1;
      dv[dcount] = odv[ocount];
      ocount++;
      }
      dcount++;
    }
    return std::make_tuple(opix, omask, dv);
}
/*** Return a pixel with only the output dates. dv contains all dates
* and only those contained in odv are kept.
*/
template <typename PixelType>
inline
PixelType  extract_output_dates(const PixelType pix, 
                                const PixelType dv, 
                                const PixelType odv)
{
  if(dv == odv)
    return pix;

  unsigned int nbDates = odv.GetSize();
  unsigned int nbInDates = dv.GetSize();

  if(nbDates > nbInDates)
      throw std::invalid_argument("There are more output dates than input dates\n");
  PixelType result{nbDates};

  unsigned int in_count = 0;
  unsigned int out_count = 0;

  while(in_count < dv.GetSize() && out_count < nbDates)
    {
    if(dv[in_count] == odv[out_count])
      {
      result[out_count] = pix[in_count];
      ++out_count;
      }
    ++in_count;
    }

  return result;
}

template <typename PixelType>
class IdentityGapFillingFunctor
{
public:
  IdentityGapFillingFunctor() = default;
  IdentityGapFillingFunctor(const PixelType& d) : dv{d} {}

  PixelType operator()(PixelType pix, PixelType mask) const
  {
    if(pix.GetSize() != mask.GetSize())
      throw std::invalid_argument("Pixel and mask have different sizes\n");
    return pix;
  }

  bool operator!=(const IdentityGapFillingFunctor a) const
  {
    return (this->dates != a.dates) || (this->dv != a.dv) ;
  }

  bool operator==(const IdentityGapFillingFunctor a) const
  {
    return !(*this != a);
  }

protected:
  PixelType dv;

};

template <typename PixelType>
class LinearGapFillingFunctor
{
public:
  using ValueType = typename PixelType::ValueType;
  using VectorType = typename std::vector<ValueType>;
  ValueType valid_value = ValueType{0};
  ValueType invalid_pixel_return_value = ValueType{0};
  LinearGapFillingFunctor() = default;
  /// Constructor with a vector of input dates
  LinearGapFillingFunctor(const PixelType& d) : dv{d} {}
  /// Constructor with vectors of input and output dates
  LinearGapFillingFunctor(const PixelType& d, const PixelType& od) 
    : dv{d}, odv{od} {}

  // valid pixel has a mask==0
  PixelType operator()(PixelType pix, PixelType mask) const
  {
    auto local_dv(dv);
    auto local_odv(odv);
    unsigned int nbDates = local_dv.GetSize();
    if(nbDates == 0) nbDates = pix.GetSize();
    if(nbDates != mask.GetSize())
      throw std::invalid_argument("Pixel and mask have different sizes\n");
    if(local_dv.GetSize()!=0 && nbDates != local_dv.GetSize())
      {
      std::stringstream  errmessg;
      errmessg << "Pixel and date vector have different sizes: " 
               << nbDates << " vs " << local_dv.GetSize() << "\n";
      throw
        std::invalid_argument(errmessg.str());
      }
    if(local_dv.GetSize()==0)
      {
      local_dv = pix;
      for(unsigned i=0; i<nbDates; i++)
        local_dv[i] = i;
      }
    if(local_odv.GetSize()==0) local_odv = local_dv;

    unsigned int nbOutputDates = local_odv.GetSize();
    PixelType validpix{nbDates};
    validpix.Fill(valid_value);
    PixelType invalidpix{nbOutputDates};
    invalidpix.Fill(invalid_pixel_return_value);
    // If the mask says all dates are valid and the input and output
    // dates are the same, keep the original value
    if(mask == validpix && local_dv == local_odv) return pix;
    // Interlace input and output dates
    PixelType tmp_pix, tmp_mask, tmp_dates;
    std::tie(tmp_pix, tmp_mask, tmp_dates) = 
      create_tmp_data_interlace_dates(pix, mask, local_dv, local_odv, valid_value);
    // For each component, find the position of the last and the next valid
    // values
    VectorType last_valid;
    VectorType next_valid;
    bool invalid_pixel;
    std::tie(last_valid, next_valid, invalid_pixel) =
      find_valid_bounds(tmp_mask, tmp_mask.GetSize(), valid_value);
    // invalid pixel?
    if(invalid_pixel)
      return invalidpix;

    return extract_output_dates(this->interpolate(tmp_pix, tmp_mask, 
                                                  tmp_dates, last_valid, 
                                                  next_valid), 
                                tmp_dates, local_odv);
  }

  bool operator!=(const LinearGapFillingFunctor a) const
  {
    return (this->dv != a.dv) ;
  }

  bool operator==(const LinearGapFillingFunctor a) const
  {
    return !(*this != a);
  }

protected:
  inline
  PixelType interpolate(const PixelType& p, const PixelType& m, const PixelType& d,
                        const VectorType& lv, const VectorType& nv) const
  {
    unsigned int nbDates = p.GetSize();
    PixelType result(nbDates);
    for(size_t i=0; i<nbDates; i++)
      {
      auto lvp = lv[i];
      auto nvp = nv[i];
      if(m[i]==valid_value)
        result[i] = p[i];
      else
        {
        // If there is no previous valid value, just use the next one
        if(lvp==-1)
          result[i] = p[nvp];
        // If there is no next valid value, just use the last one
        else if(nvp==nbDates)
          result[i] = p[lvp];
        // Otherwise, use linear interpolation
        else
          {
          double x1 = d[lvp];
          double y1 = p[lvp];
          double x2 = d[nvp];
          double y2 = p[nvp];
          double a = (y2-y1)/(x2-x1);
          double b = ((y1+y2)*(x2-x1)-(y2-y1)*(x2+x1))/(2*(x2-x1));

          result[i] = a*d[i]+b;
          }
        }
      }
    return result;
  }
  
  /// Input date vector
  PixelType dv;
  /// Output date vector
  PixelType odv;

};

template <typename PixelType>
class SplineGapFillingFunctor
{
public:
  using ValueType = typename PixelType::ValueType;
  using VectorType = typename std::vector<ValueType>;
  ValueType valid_value = ValueType{0};
  ValueType invalid_pixel_return_value = ValueType{0};
  SplineGapFillingFunctor() = default;

  SplineGapFillingFunctor(const PixelType& d) : dv{d} {}

  /// Constructor with vectors of input and output dates
  SplineGapFillingFunctor(const PixelType& d, const PixelType& od) 
    : dv{d}, odv{od} {}


  // valid pixel has a mask==0
  PixelType operator()(PixelType pix, PixelType mask) const
  {
    auto local_dv(dv);
    auto local_odv(odv);
    unsigned int nbDates = local_dv.GetSize();
    if(nbDates == 0) nbDates = pix.GetSize();
    if(nbDates != mask.GetSize())
      throw std::invalid_argument("Pixel and mask have different sizes\n");
    if(local_dv.GetSize()!=0 && nbDates != local_dv.GetSize())
      {
      std::stringstream  errmessg;
      errmessg << "Pixel and date vector have different sizes: " 
               << nbDates << " vs " << local_dv.GetSize() << "\n";
      throw
        std::invalid_argument(errmessg.str());
      }
    if(local_dv.GetSize()==0)
      {
      local_dv = pix;
      for(size_t i=0; i<nbDates; i++)
        local_dv[i] = i;
      }
    if(local_odv.GetSize()==0) local_odv = local_dv;

    unsigned int nbOutputDates = local_odv.GetSize();
    PixelType validpix{nbDates};
    validpix.Fill(valid_value);
    PixelType invalidpix{nbOutputDates};
    invalidpix.Fill(invalid_pixel_return_value);
    // If the mask says all dates are valid and the input and output
    // dates are the same, keep the original value
    if(mask == validpix && local_dv == local_odv) return pix;
    // Interlace input and output dates
    PixelType tmp_pix, tmp_mask, tmp_dates;
    std::tie(tmp_pix, tmp_mask, tmp_dates) = 
      create_tmp_data_interlace_dates(pix, mask, local_dv, local_odv, valid_value);
    // For each component, find the position of the last and the next valid
    // values
    VectorType last_valid;
    VectorType next_valid;
    bool invalid_pixel;
    std::tie(last_valid, next_valid, invalid_pixel) =
      find_valid_bounds(tmp_mask, tmp_mask.GetSize(), valid_value);
    // invalid pixel?
    if(invalid_pixel)
      return invalidpix;

    return extract_output_dates(this->interpolate(tmp_pix, tmp_mask, 
                                                  tmp_dates, last_valid, 
                                                  next_valid), 
                                tmp_dates, local_odv);
  }

  bool operator!=(const SplineGapFillingFunctor a) const
  {
    return (this->dv != a.dv) ;
  }

  bool operator==(const SplineGapFillingFunctor a) const
  {
    return !(*this != a);
  }

protected:
  inline
  PixelType interpolate(const PixelType& p, const PixelType& m, 
                        const PixelType& d, const VectorType& lv, 
                        const VectorType& nv) const
  {
    unsigned int nbDates = p.GetSize();
    // Prepare the data for gsl
    double* x = new double[nbDates];
    double* y = new double[nbDates];
    std::size_t nbValidDates{0};
    for(size_t i = 0; i < nbDates; i++)
      {
      if(m[i]==valid_value)
        {
        x[nbValidDates] = d[i];
        //Ensure that the dates are strictly increasing
        if(nbValidDates>0 && 
           (x[nbValidDates]-x[nbValidDates-1])<std::numeric_limits<double>::epsilon())
        {
          //1 epsilon is not enough for gsl
          x[nbValidDates] = x[nbValidDates-1] + 0.1;
        }
        y[nbValidDates] = p[i];
        nbValidDates++;
        }
      }
    if(nbValidDates < 2)
    {
    return p;
    }
    gsl_interp_accel* acc = gsl_interp_accel_alloc();
    gsl_spline* spline = select_spline_type(nbValidDates);
    if(spline == nullptr) return p;
    gsl_spline_init(spline, x, y, nbValidDates);
    // the real interpolation
    PixelType result(nbDates);
    for(size_t i=0; i<nbDates; i++)
         {
         auto lvp = lv[i];
         auto nvp = nv[i];
         if(m[i]==(valid_value))
           result[i] = p[i];
         else
           {
           // If there is no previous valid value, just use the next one
           if(lvp==-1)
             result[i] = p[nvp];
           // If there is no next valid value, just use the last one
           else if(nvp==nbDates)
             result[i] = p[lvp];
           // Otherwise, use spline interpolation
           else
             {
             result[i] = gsl_spline_eval(spline, d[i], acc);
             }
           }
         }
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    delete [] x;
    delete [] y;
    return result;
  }

  gsl_spline* select_spline_type(std::size_t nbDates) const
  {
    switch(nbDates)
      {
      case 0:
      case 1:
        return nullptr;
      case 2:
        return gsl_spline_alloc(gsl_interp_linear, nbDates);
        break;
      case 3:
      case 4:
        return gsl_spline_alloc(gsl_interp_cspline, nbDates);
        break;
      default:
        return gsl_spline_alloc(gsl_interp_akima, nbDates);
      }
  }
/// Input date vector
  PixelType dv;
/// Output date vector
  PixelType odv;
};

/**
Adapts a functor operating on time series so that it can work with
series which have several components per date. The p2 pixel can have
just one component per date.
*/
template <typename PixelType, typename FunctorType>
struct MultiComponentTimeSeriesFunctorAdaptor {
  MultiComponentTimeSeriesFunctorAdaptor() : m_NumberOfComponentsPerDate(1),
                                             m_MaxNumberOfOutputDates(20){};
  PixelType operator()(PixelType p1, PixelType p2) const
  {
    const auto nbComponents = p1.GetSize();
    const auto nbDates = nbComponents/m_NumberOfComponentsPerDate;
    CheckSizes(p1, p2);
    // Due to date interlacing, we don't know here the size of the
    // output pixel. This is only known when we receive the result
    // from the functor. The worst case is all dates are dulicated
    PixelType result(nbComponents*m_MaxNumberOfOutputDates);
    unsigned int outNbDates{0};
    for(size_t band=0; band<m_NumberOfComponentsPerDate; band++)
      {
      auto tmp1 = PixelType(nbDates);
      auto tmp2 = PixelType(nbDates);
      for(size_t date=0; date<nbDates; date++)
        tmp1[date] = p1[band+date*m_NumberOfComponentsPerDate];

      PixelType tmp_res;
      // If p1 and p2 have the same sizes, demux also p2
      if(p1.GetSize() == p2.GetSize())
        {
        for(size_t date=0; date<nbDates; date++)
          tmp2[date] = p2[band+date*m_NumberOfComponentsPerDate];
        tmp_res = m_Functor(tmp1, tmp2);
        }
      // Otherwise, use the same p2 for all components of p1
      else
        tmp_res = m_Functor(tmp1, p2);

      outNbDates = tmp_res.GetSize();
      if(outNbDates > result.GetSize())
        {
        std::stringstream errmessg;
        errmessg << "The result pixel has too many components: "
                 <<  outNbDates << " instead of expected max of "
                 << result.GetSize() << std::endl;
        throw 
          std::invalid_argument(errmessg.str());
        }
      for(size_t date=0; date<outNbDates; date++)
        result[band+date*m_NumberOfComponentsPerDate] = tmp_res[date];
      }

    auto output_size = outNbDates*m_NumberOfComponentsPerDate;
    //resize keeping only the front values
    result.SetSize(output_size);
    return result;
  }
  void CheckSizes(PixelType p1, PixelType p2) const
  {
    const auto nbComponents = p1.GetSize();
    const auto nbDates = nbComponents/m_NumberOfComponentsPerDate;
    if(nbComponents < m_NumberOfComponentsPerDate)
      {
      std::stringstream errmessg;
      errmessg << "Using " << m_NumberOfComponentsPerDate 
               << " components per date, but pixel has only "
               << nbComponents << "\n";
      throw
        std::invalid_argument(errmessg.str());
      }
    if(p1.GetSize()!=p2.GetSize() && p2.GetSize()!=nbDates)
      {
      std::stringstream errmessg;
      errmessg << "p2 has to have either the same size as p1 "
               << "or one component per date\n" 
               << "p1 is " << p1.GetSize() << "\n"
               << "p2 is " << p2.GetSize() << "\n"
               << "nbDates is " << nbDates << "\n";
      throw
        std::invalid_argument(errmessg.str());
      }
  }
  void SetNumberOfComponentsPerDate(size_t n)
  {
    m_NumberOfComponentsPerDate = n;
  }
  void SetMaxNumberOfOutputDates(size_t n)
  {
    m_MaxNumberOfOutputDates = n;
  }
  void SetFunctor(FunctorType f)
  {
    m_Functor = f;
  }
  FunctorType* GetFunctor() const
  {
    return &m_Functor;
  }
private:
  size_t m_NumberOfComponentsPerDate;
  size_t m_MaxNumberOfOutputDates;
  FunctorType m_Functor;
};

/**
The gapfill_time_series function takes 2 input time series files (the
image data and the mask data) and produces an output file (the
gapfilled image data). The number of components (spectral bands, for
instance) per date can be provided (default is 1) so that the
different components are processed as individual time series. A file
containing the dates of the acquisition can be provided. It should
contain one date per line in YYYMMDD format.

The mask time series is supposed to contain 0s for the valid dates.
The function is templated over the ImageType (typically an
otb::VectorImage<double,2>) and on the interpolating functor.
*/
template <typename ImageType, typename FunctorType, 
          typename MultiComponentFunctorType =
          MultiComponentTimeSeriesFunctorAdaptor<typename ImageType::PixelType,
                                                 FunctorType>>
void gapfill_time_series(typename ImageType::Pointer inIma, 
                         typename ImageType::Pointer maskIma, 
                         typename BinaryFunctorImageFilterWithNBands<ImageType, 
                         ImageType, ImageType, FunctorType>::Pointer filter,
                         typename BinaryFunctorImageFilterWithNBands<ImageType, 
                         ImageType, ImageType,
                         MultiComponentFunctorType>::Pointer filter_mc,
                         size_t components_per_date = 1, 
                         typename ImageType::PixelType dv= typename ImageType::PixelType{},
                         typename ImageType::PixelType odv= typename ImageType::PixelType{})
{

  inIma->UpdateOutputInformation();
  maskIma->UpdateOutputInformation();

  using TPixel = typename ImageType::PixelType;

  auto number_of_input_components = inIma->GetNumberOfComponentsPerPixel();
  unsigned int number_of_output_components{number_of_input_components};
  if(dv != TPixel{})
    {
    number_of_output_components = components_per_date*dv.GetSize();
    }
  if(odv != TPixel{})
    {
    number_of_output_components = components_per_date*odv.GetSize();
    }

  filter->SetNumberOfOutputBands(number_of_output_components);
  
  filter_mc->SetNumberOfOutputBands(number_of_output_components);

  if(components_per_date==1)
    {
    if(odv != TPixel{})
      filter->SetFunctor(FunctorType(dv, odv));
    else if(dv != TPixel{})
      filter->SetFunctor(FunctorType(dv));
    filter->SetInput(0, inIma);
    filter->SetInput(1, maskIma);
    filter->UpdateOutputInformation();
    }
  else
    {
    if(odv != TPixel{})
      {
      (filter_mc->GetFunctor()).SetFunctor(FunctorType(dv, odv));
      (filter_mc->GetFunctor()).SetMaxNumberOfOutputDates(odv.GetSize());
      }
    else if(dv != TPixel{})
      (filter_mc->GetFunctor()).SetFunctor(FunctorType(dv));
    (filter_mc->GetFunctor()).SetNumberOfComponentsPerDate(components_per_date);
    filter_mc->SetInput(0, inIma);
    filter_mc->SetInput(1, maskIma);
    filter_mc->UpdateOutputInformation();
    }
}

}//GapFilling namespace

#endif

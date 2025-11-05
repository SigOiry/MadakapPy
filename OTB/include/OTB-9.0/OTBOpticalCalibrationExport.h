
#ifndef OTBOpticalCalibration_EXPORT_H
#define OTBOpticalCalibration_EXPORT_H

#ifdef OTB_STATIC
#  define OTBOpticalCalibration_EXPORT
#  define OTBOpticalCalibration_HIDDEN
#  define OTBOpticalCalibration_EXPORT_TEMPLATE
#  define OTBOpticalCalibration_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBOpticalCalibration_EXPORT
#    ifdef OTBOpticalCalibration_EXPORTS
        /* We are building this library */
#      define OTBOpticalCalibration_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBOpticalCalibration_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBOpticalCalibration_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBOpticalCalibration_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBOpticalCalibration_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBOpticalCalibration_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBOpticalCalibration_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBOpticalCalibration_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBOpticalCalibration_HIDDEN
#    define OTBOpticalCalibration_HIDDEN 
#  endif
#endif

#ifndef OTBOpticalCalibration_DEPRECATED
#  define OTBOpticalCalibration_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBOpticalCalibration_DEPRECATED_EXPORT
#  define OTBOpticalCalibration_DEPRECATED_EXPORT OTBOpticalCalibration_EXPORT OTBOpticalCalibration_DEPRECATED
#endif

#ifndef OTBOpticalCalibration_DEPRECATED_NO_EXPORT
#  define OTBOpticalCalibration_DEPRECATED_NO_EXPORT OTBOpticalCalibration_HIDDEN OTBOpticalCalibration_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBOPTICALCALIBRATION_NO_DEPRECATED
#    define OTBOPTICALCALIBRATION_NO_DEPRECATED
#  endif
#endif

#endif /* OTBOpticalCalibration_EXPORT_H */


#ifndef OTBSampling_EXPORT_H
#define OTBSampling_EXPORT_H

#ifdef OTB_STATIC
#  define OTBSampling_EXPORT
#  define OTBSampling_HIDDEN
#  define OTBSampling_EXPORT_TEMPLATE
#  define OTBSampling_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBSampling_EXPORT
#    ifdef OTBSampling_EXPORTS
        /* We are building this library */
#      define OTBSampling_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBSampling_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBSampling_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBSampling_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBSampling_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBSampling_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBSampling_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBSampling_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBSampling_HIDDEN
#    define OTBSampling_HIDDEN 
#  endif
#endif

#ifndef OTBSampling_DEPRECATED
#  define OTBSampling_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBSampling_DEPRECATED_EXPORT
#  define OTBSampling_DEPRECATED_EXPORT OTBSampling_EXPORT OTBSampling_DEPRECATED
#endif

#ifndef OTBSampling_DEPRECATED_NO_EXPORT
#  define OTBSampling_DEPRECATED_NO_EXPORT OTBSampling_HIDDEN OTBSampling_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBSAMPLING_NO_DEPRECATED
#    define OTBSAMPLING_NO_DEPRECATED
#  endif
#endif

#endif /* OTBSampling_EXPORT_H */

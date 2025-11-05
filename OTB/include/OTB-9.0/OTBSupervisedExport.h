
#ifndef OTBSupervised_EXPORT_H
#define OTBSupervised_EXPORT_H

#ifdef OTB_STATIC
#  define OTBSupervised_EXPORT
#  define OTBSupervised_HIDDEN
#  define OTBSupervised_EXPORT_TEMPLATE
#  define OTBSupervised_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBSupervised_EXPORT
#    ifdef OTBSupervised_EXPORTS
        /* We are building this library */
#      define OTBSupervised_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBSupervised_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBSupervised_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBSupervised_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBSupervised_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBSupervised_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBSupervised_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBSupervised_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBSupervised_HIDDEN
#    define OTBSupervised_HIDDEN 
#  endif
#endif

#ifndef OTBSupervised_DEPRECATED
#  define OTBSupervised_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBSupervised_DEPRECATED_EXPORT
#  define OTBSupervised_DEPRECATED_EXPORT OTBSupervised_EXPORT OTBSupervised_DEPRECATED
#endif

#ifndef OTBSupervised_DEPRECATED_NO_EXPORT
#  define OTBSupervised_DEPRECATED_NO_EXPORT OTBSupervised_HIDDEN OTBSupervised_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBSUPERVISED_NO_DEPRECATED
#    define OTBSUPERVISED_NO_DEPRECATED
#  endif
#endif

#endif /* OTBSupervised_EXPORT_H */


#ifndef OTBGdalAdapters_EXPORT_H
#define OTBGdalAdapters_EXPORT_H

#ifdef OTB_STATIC
#  define OTBGdalAdapters_EXPORT
#  define OTBGdalAdapters_HIDDEN
#  define OTBGdalAdapters_EXPORT_TEMPLATE
#  define OTBGdalAdapters_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBGdalAdapters_EXPORT
#    ifdef OTBGdalAdapters_EXPORTS
        /* We are building this library */
#      define OTBGdalAdapters_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBGdalAdapters_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBGdalAdapters_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBGdalAdapters_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBGdalAdapters_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBGdalAdapters_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBGdalAdapters_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBGdalAdapters_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBGdalAdapters_HIDDEN
#    define OTBGdalAdapters_HIDDEN 
#  endif
#endif

#ifndef OTBGdalAdapters_DEPRECATED
#  define OTBGdalAdapters_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBGdalAdapters_DEPRECATED_EXPORT
#  define OTBGdalAdapters_DEPRECATED_EXPORT OTBGdalAdapters_EXPORT OTBGdalAdapters_DEPRECATED
#endif

#ifndef OTBGdalAdapters_DEPRECATED_NO_EXPORT
#  define OTBGdalAdapters_DEPRECATED_NO_EXPORT OTBGdalAdapters_HIDDEN OTBGdalAdapters_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBGDALADAPTERS_NO_DEPRECATED
#    define OTBGDALADAPTERS_NO_DEPRECATED
#  endif
#endif

#endif /* OTBGdalAdapters_EXPORT_H */

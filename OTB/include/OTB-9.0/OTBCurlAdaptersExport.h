
#ifndef OTBCurlAdapters_EXPORT_H
#define OTBCurlAdapters_EXPORT_H

#ifdef OTB_STATIC
#  define OTBCurlAdapters_EXPORT
#  define OTBCurlAdapters_HIDDEN
#  define OTBCurlAdapters_EXPORT_TEMPLATE
#  define OTBCurlAdapters_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBCurlAdapters_EXPORT
#    ifdef OTBCurlAdapters_EXPORTS
        /* We are building this library */
#      define OTBCurlAdapters_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBCurlAdapters_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBCurlAdapters_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBCurlAdapters_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBCurlAdapters_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBCurlAdapters_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBCurlAdapters_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBCurlAdapters_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBCurlAdapters_HIDDEN
#    define OTBCurlAdapters_HIDDEN 
#  endif
#endif

#ifndef OTBCurlAdapters_DEPRECATED
#  define OTBCurlAdapters_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBCurlAdapters_DEPRECATED_EXPORT
#  define OTBCurlAdapters_DEPRECATED_EXPORT OTBCurlAdapters_EXPORT OTBCurlAdapters_DEPRECATED
#endif

#ifndef OTBCurlAdapters_DEPRECATED_NO_EXPORT
#  define OTBCurlAdapters_DEPRECATED_NO_EXPORT OTBCurlAdapters_HIDDEN OTBCurlAdapters_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBCURLADAPTERS_NO_DEPRECATED
#    define OTBCURLADAPTERS_NO_DEPRECATED
#  endif
#endif

#endif /* OTBCurlAdapters_EXPORT_H */

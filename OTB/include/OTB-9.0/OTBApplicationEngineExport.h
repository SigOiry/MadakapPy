
#ifndef OTBApplicationEngine_EXPORT_H
#define OTBApplicationEngine_EXPORT_H

#ifdef OTB_STATIC
#  define OTBApplicationEngine_EXPORT
#  define OTBApplicationEngine_HIDDEN
#  define OTBApplicationEngine_EXPORT_TEMPLATE
#  define OTBApplicationEngine_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBApplicationEngine_EXPORT
#    ifdef OTBApplicationEngine_EXPORTS
        /* We are building this library */
#      define OTBApplicationEngine_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBApplicationEngine_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBApplicationEngine_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBApplicationEngine_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBApplicationEngine_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBApplicationEngine_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBApplicationEngine_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBApplicationEngine_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBApplicationEngine_HIDDEN
#    define OTBApplicationEngine_HIDDEN 
#  endif
#endif

#ifndef OTBApplicationEngine_DEPRECATED
#  define OTBApplicationEngine_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBApplicationEngine_DEPRECATED_EXPORT
#  define OTBApplicationEngine_DEPRECATED_EXPORT OTBApplicationEngine_EXPORT OTBApplicationEngine_DEPRECATED
#endif

#ifndef OTBApplicationEngine_DEPRECATED_NO_EXPORT
#  define OTBApplicationEngine_DEPRECATED_NO_EXPORT OTBApplicationEngine_HIDDEN OTBApplicationEngine_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBAPPLICATIONENGINE_NO_DEPRECATED
#    define OTBAPPLICATIONENGINE_NO_DEPRECATED
#  endif
#endif

#endif /* OTBApplicationEngine_EXPORT_H */

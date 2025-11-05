
#ifndef OTBCommon_EXPORT_H
#define OTBCommon_EXPORT_H

#ifdef OTB_STATIC
#  define OTBCommon_EXPORT
#  define OTBCommon_HIDDEN
#  define OTBCommon_EXPORT_TEMPLATE
#  define OTBCommon_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBCommon_EXPORT
#    ifdef OTBCommon_EXPORTS
        /* We are building this library */
#      define OTBCommon_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBCommon_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBCommon_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBCommon_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBCommon_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBCommon_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBCommon_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBCommon_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBCommon_HIDDEN
#    define OTBCommon_HIDDEN 
#  endif
#endif

#ifndef OTBCommon_DEPRECATED
#  define OTBCommon_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBCommon_DEPRECATED_EXPORT
#  define OTBCommon_DEPRECATED_EXPORT OTBCommon_EXPORT OTBCommon_DEPRECATED
#endif

#ifndef OTBCommon_DEPRECATED_NO_EXPORT
#  define OTBCommon_DEPRECATED_NO_EXPORT OTBCommon_HIDDEN OTBCommon_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBCOMMON_NO_DEPRECATED
#    define OTBCOMMON_NO_DEPRECATED
#  endif
#endif

#endif /* OTBCommon_EXPORT_H */

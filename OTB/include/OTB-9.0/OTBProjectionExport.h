
#ifndef OTBProjection_EXPORT_H
#define OTBProjection_EXPORT_H

#ifdef OTB_STATIC
#  define OTBProjection_EXPORT
#  define OTBProjection_HIDDEN
#  define OTBProjection_EXPORT_TEMPLATE
#  define OTBProjection_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBProjection_EXPORT
#    ifdef OTBProjection_EXPORTS
        /* We are building this library */
#      define OTBProjection_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBProjection_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBProjection_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBProjection_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBProjection_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBProjection_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBProjection_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBProjection_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBProjection_HIDDEN
#    define OTBProjection_HIDDEN 
#  endif
#endif

#ifndef OTBProjection_DEPRECATED
#  define OTBProjection_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBProjection_DEPRECATED_EXPORT
#  define OTBProjection_DEPRECATED_EXPORT OTBProjection_EXPORT OTBProjection_DEPRECATED
#endif

#ifndef OTBProjection_DEPRECATED_NO_EXPORT
#  define OTBProjection_DEPRECATED_NO_EXPORT OTBProjection_HIDDEN OTBProjection_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBPROJECTION_NO_DEPRECATED
#    define OTBPROJECTION_NO_DEPRECATED
#  endif
#endif

#endif /* OTBProjection_EXPORT_H */

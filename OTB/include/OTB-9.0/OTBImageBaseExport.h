
#ifndef OTBImageBase_EXPORT_H
#define OTBImageBase_EXPORT_H

#ifdef OTB_STATIC
#  define OTBImageBase_EXPORT
#  define OTBImageBase_HIDDEN
#  define OTBImageBase_EXPORT_TEMPLATE
#  define OTBImageBase_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBImageBase_EXPORT
#    ifdef OTBImageBase_EXPORTS
        /* We are building this library */
#      define OTBImageBase_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBImageBase_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBImageBase_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBImageBase_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBImageBase_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBImageBase_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBImageBase_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBImageBase_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBImageBase_HIDDEN
#    define OTBImageBase_HIDDEN 
#  endif
#endif

#ifndef OTBImageBase_DEPRECATED
#  define OTBImageBase_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBImageBase_DEPRECATED_EXPORT
#  define OTBImageBase_DEPRECATED_EXPORT OTBImageBase_EXPORT OTBImageBase_DEPRECATED
#endif

#ifndef OTBImageBase_DEPRECATED_NO_EXPORT
#  define OTBImageBase_DEPRECATED_NO_EXPORT OTBImageBase_HIDDEN OTBImageBase_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBIMAGEBASE_NO_DEPRECATED
#    define OTBIMAGEBASE_NO_DEPRECATED
#  endif
#endif

#endif /* OTBImageBase_EXPORT_H */

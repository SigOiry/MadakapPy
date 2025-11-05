
#ifndef OTBMetadata_EXPORT_H
#define OTBMetadata_EXPORT_H

#ifdef OTB_STATIC
#  define OTBMetadata_EXPORT
#  define OTBMetadata_HIDDEN
#  define OTBMetadata_EXPORT_TEMPLATE
#  define OTBMetadata_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBMetadata_EXPORT
#    ifdef OTBMetadata_EXPORTS
        /* We are building this library */
#      define OTBMetadata_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBMetadata_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBMetadata_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBMetadata_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBMetadata_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBMetadata_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBMetadata_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBMetadata_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBMetadata_HIDDEN
#    define OTBMetadata_HIDDEN 
#  endif
#endif

#ifndef OTBMetadata_DEPRECATED
#  define OTBMetadata_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBMetadata_DEPRECATED_EXPORT
#  define OTBMetadata_DEPRECATED_EXPORT OTBMetadata_EXPORT OTBMetadata_DEPRECATED
#endif

#ifndef OTBMetadata_DEPRECATED_NO_EXPORT
#  define OTBMetadata_DEPRECATED_NO_EXPORT OTBMetadata_HIDDEN OTBMetadata_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBMETADATA_NO_DEPRECATED
#    define OTBMETADATA_NO_DEPRECATED
#  endif
#endif

#endif /* OTBMetadata_EXPORT_H */

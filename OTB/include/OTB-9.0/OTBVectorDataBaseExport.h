
#ifndef OTBVectorDataBase_EXPORT_H
#define OTBVectorDataBase_EXPORT_H

#ifdef OTB_STATIC
#  define OTBVectorDataBase_EXPORT
#  define OTBVectorDataBase_HIDDEN
#  define OTBVectorDataBase_EXPORT_TEMPLATE
#  define OTBVectorDataBase_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBVectorDataBase_EXPORT
#    ifdef OTBVectorDataBase_EXPORTS
        /* We are building this library */
#      define OTBVectorDataBase_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBVectorDataBase_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBVectorDataBase_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBVectorDataBase_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBVectorDataBase_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBVectorDataBase_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBVectorDataBase_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBVectorDataBase_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBVectorDataBase_HIDDEN
#    define OTBVectorDataBase_HIDDEN 
#  endif
#endif

#ifndef OTBVectorDataBase_DEPRECATED
#  define OTBVectorDataBase_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBVectorDataBase_DEPRECATED_EXPORT
#  define OTBVectorDataBase_DEPRECATED_EXPORT OTBVectorDataBase_EXPORT OTBVectorDataBase_DEPRECATED
#endif

#ifndef OTBVectorDataBase_DEPRECATED_NO_EXPORT
#  define OTBVectorDataBase_DEPRECATED_NO_EXPORT OTBVectorDataBase_HIDDEN OTBVectorDataBase_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBVECTORDATABASE_NO_DEPRECATED
#    define OTBVECTORDATABASE_NO_DEPRECATED
#  endif
#endif

#endif /* OTBVectorDataBase_EXPORT_H */


#ifndef OTBLearningBase_EXPORT_H
#define OTBLearningBase_EXPORT_H

#ifdef OTB_STATIC
#  define OTBLearningBase_EXPORT
#  define OTBLearningBase_HIDDEN
#  define OTBLearningBase_EXPORT_TEMPLATE
#  define OTBLearningBase_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBLearningBase_EXPORT
#    ifdef OTBLearningBase_EXPORTS
        /* We are building this library */
#      define OTBLearningBase_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBLearningBase_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBLearningBase_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBLearningBase_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBLearningBase_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBLearningBase_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBLearningBase_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBLearningBase_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBLearningBase_HIDDEN
#    define OTBLearningBase_HIDDEN 
#  endif
#endif

#ifndef OTBLearningBase_DEPRECATED
#  define OTBLearningBase_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBLearningBase_DEPRECATED_EXPORT
#  define OTBLearningBase_DEPRECATED_EXPORT OTBLearningBase_EXPORT OTBLearningBase_DEPRECATED
#endif

#ifndef OTBLearningBase_DEPRECATED_NO_EXPORT
#  define OTBLearningBase_DEPRECATED_NO_EXPORT OTBLearningBase_HIDDEN OTBLearningBase_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBLEARNINGBASE_NO_DEPRECATED
#    define OTBLEARNINGBASE_NO_DEPRECATED
#  endif
#endif

#endif /* OTBLearningBase_EXPORT_H */

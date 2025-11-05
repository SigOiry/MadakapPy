
#ifndef OTBSimulation_EXPORT_H
#define OTBSimulation_EXPORT_H

#ifdef OTB_STATIC
#  define OTBSimulation_EXPORT
#  define OTBSimulation_HIDDEN
#  define OTBSimulation_EXPORT_TEMPLATE
#  define OTBSimulation_EXPORT_EXPLICIT_TEMPLATE
#else
#  ifndef OTBSimulation_EXPORT
#    ifdef OTBSimulation_EXPORTS
        /* We are building this library */
#      define OTBSimulation_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBSimulation_EXPORT __declspec(dllimport)
#    endif
#  endif
#  ifndef OTBSimulation_EXPORT_TEMPLATE
        /* We are building this library */
#      define OTBSimulation_EXPORT_TEMPLATE 
#    else
        /* We are using this library */
#      define OTBSimulation_EXPORT_TEMPLATE 
#  endif
#  ifndef OTBSimulation_EXPORT_EXPLICIT_TEMPLATE
        /* We are building this library */
#      define OTBSimulation_EXPORT_EXPLICIT_TEMPLATE __declspec(dllexport)
#    else
        /* We are using this library */
#      define OTBSimulation_EXPORT_EXPLICIT_TEMPLATE __declspec(dllimport)
#  endif
#  ifndef OTBSimulation_HIDDEN
#    define OTBSimulation_HIDDEN 
#  endif
#endif

#ifndef OTBSimulation_DEPRECATED
#  define OTBSimulation_DEPRECATED __declspec(deprecated)
#endif

#ifndef OTBSimulation_DEPRECATED_EXPORT
#  define OTBSimulation_DEPRECATED_EXPORT OTBSimulation_EXPORT OTBSimulation_DEPRECATED
#endif

#ifndef OTBSimulation_DEPRECATED_NO_EXPORT
#  define OTBSimulation_DEPRECATED_NO_EXPORT OTBSimulation_HIDDEN OTBSimulation_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OTBSIMULATION_NO_DEPRECATED
#    define OTBSIMULATION_NO_DEPRECATED
#  endif
#endif

#endif /* OTBSimulation_EXPORT_H */

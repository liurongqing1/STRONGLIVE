
#ifndef TC_CORE_EXPORT_H
#define TC_CORE_EXPORT_H

#ifdef TC_CORE_STATIC_DEFINE
#  define TC_CORE_EXPORT
#  define TC_CORE_NO_EXPORT
#else
#  ifndef TC_CORE_EXPORT
#    ifdef TC_CORE_EXPORTS
        /* We are building this library */
#      define TC_CORE_EXPORT 
#    else
        /* We are using this library */
#      define TC_CORE_EXPORT 
#    endif
#  endif

#  ifndef TC_CORE_NO_EXPORT
#    define TC_CORE_NO_EXPORT 
#  endif
#endif

#ifndef TC_CORE_DEPRECATED
#  define TC_CORE_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef TC_CORE_DEPRECATED_EXPORT
#  define TC_CORE_DEPRECATED_EXPORT TC_CORE_EXPORT TC_CORE_DEPRECATED
#endif

#ifndef TC_CORE_DEPRECATED_NO_EXPORT
#  define TC_CORE_DEPRECATED_NO_EXPORT TC_CORE_NO_EXPORT TC_CORE_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef TC_CORE_NO_DEPRECATED
#    define TC_CORE_NO_DEPRECATED
#  endif
#endif

#endif /* TC_CORE_EXPORT_H */

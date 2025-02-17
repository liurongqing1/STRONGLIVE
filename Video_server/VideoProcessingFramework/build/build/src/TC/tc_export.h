
#ifndef TC_EXPORT_H
#define TC_EXPORT_H

#ifdef TC_STATIC_DEFINE
#  define TC_EXPORT
#  define TC_NO_EXPORT
#else
#  ifndef TC_EXPORT
#    ifdef TC_EXPORTS
        /* We are building this library */
#      define TC_EXPORT 
#    else
        /* We are using this library */
#      define TC_EXPORT 
#    endif
#  endif

#  ifndef TC_NO_EXPORT
#    define TC_NO_EXPORT 
#  endif
#endif

#ifndef TC_DEPRECATED
#  define TC_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef TC_DEPRECATED_EXPORT
#  define TC_DEPRECATED_EXPORT TC_EXPORT TC_DEPRECATED
#endif

#ifndef TC_DEPRECATED_NO_EXPORT
#  define TC_DEPRECATED_NO_EXPORT TC_NO_EXPORT TC_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef TC_NO_DEPRECATED
#    define TC_NO_DEPRECATED
#  endif
#endif

#endif /* TC_EXPORT_H */


cdef extern from "hsluv.h" nogil:
    
    void hsluv2rgb(double   h, double   s, double   l,
                   double* pr, double* pg, double* pb)
    
    void rgb2hsluv(double   r, double   g, double   b,
                   double* ph, double* ps, double* pl)
    
    void hpluv2rgb(double   h, double   s, double   l,
                   double* pr, double* pg, double* pb)
    
    void rgb2hpluv(double   r, double   g, double   b,
                   double* ph, double* ps, double* pl)

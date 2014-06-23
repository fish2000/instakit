from libc.stdint cimport uint32_t

# cdef extern from "hqx/src/common.h":
#     pass

cdef extern from "hqx/src/hqx.h":
    
    # void hqxInit()
    
    void hq2x_32(
        uint32_t* src,
        uint32_t* dest,
        int width, int height)
    
    void hq3x_32(
        uint32_t* src,
        uint32_t* dest,
        int width, int height)
    
    void hq4x_32(
        uint32_t* src,
        uint32_t* dest,
        int width, int height)
    
    void hq2x_32_rb(
        uint32_t* src, uint32_t src_bytes,
        uint32_t* dest, uint32_t dest_bytes,
        int width, int height)
    
    void hq3x_32_rb(
        uint32_t* src, uint32_t src_bytes,
        uint32_t* dest, uint32_t dest_bytes,
        int width, int height)
    
    void hq4x_32_rb(
        uint32_t* src, uint32_t src_bytes,
        uint32_t* dest, uint32_t dest_bytes,
        int width, int height)

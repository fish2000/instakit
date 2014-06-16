#ifndef adderror
    #define adderror( b, e ) ( ((b + e) <= 0x00) ? 0x00 : ( (( b + e) >= 0xFF) ? 0xFF : (b + e) ) )
#endif

unsigned char threshold_matrix[256];

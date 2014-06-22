lib_LTLIBRARIES = libhqx.la
libhqx_la_LDFLAGS = -version-info @HQX_LTVERSION@
if WINDOWS_BUILD
    libhqx_la_LDFLAGS += -no-undefined
endif
libhqx_la_SOURCES = init.c hq2x.c hq3x.c hq4x.c common.h
include_HEADERS = hqx.h

bin_PROGRAMS = hqx
hqx_SOURCES = hqx.c
hqx_CPPFLAGS = -I.
hqx_LDFLAGS = -lIL libhqx.la

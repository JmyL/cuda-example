find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    # https://ccache.dev/manual/4.9.1.html#_precompiled_headers
    set(CMAKE_C_COMPILER_LAUNCHER
        "${CCACHE_PROGRAM};namespace=uh-core;sloppiness=pch_defines,time_macros"
    )
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CMAKE_C_COMPILER_LAUNCHER})
endif()
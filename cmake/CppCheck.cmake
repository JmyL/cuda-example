function(add_cppcheck target)
    find_program(CPPCHECK_PATH cppcheck REQUIRED)
    set_target_properties(
        ${target}
        PROPERTIES
            CXX_CPPCHECK
                "${CPPCHECK_PATH};--enable=style,performance,portability;--error-exitcode=10;--std=c++${CMAKE_CXX_STANDARD};--suppress=missingIncludeSystem"
    )
endfunction()
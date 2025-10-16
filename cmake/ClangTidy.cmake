function(add_clangtidy target)
    find_program(CLANGTIDY_PATH clang-tidy REQUIRED)
    set_target_properties(
        ${target}
        PROPERTIES
            CXX_CLANGTIDY
                "${CLANGTIDY_PATH};-checks=-*,performance-*,modernize-*,portability-*"
    )
endfunction()
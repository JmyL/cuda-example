function(apply_stl_debug_check target)
    target_compile_definitions(
        ${target}
        PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_DEBUG>
    )
endfunction()

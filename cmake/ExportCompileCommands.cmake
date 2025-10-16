set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_custom_target(
    copy-compile-commands
    ALL
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}
)
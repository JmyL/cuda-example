cmake_minimum_required(VERSION 3.20.0)
cmake_policy(SET CMP0167 NEW)

project(Calc VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(EnableCUDA)
include(ExportCompileCommands)
include(NoInSourceBuilds)
include(SetReleaseAsDefault)
include(UseCcache)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src bin)
include(CTest)
add_subdirectory(test)
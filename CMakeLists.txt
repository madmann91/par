cmake_minimum_required(VERSION 3.9)
project(par LANGUAGES CXX VERSION 1.0)

add_library(par INTERFACE)
target_include_directories(par INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

include(CMakePackageConfigHelpers)

add_library(par::par ALIAS par)

export(TARGETS par NAMESPACE par:: FILE par-config.cmake)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/par-config-version.cmake COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/par-config-version.cmake DESTINATION lib/cmake)
install(DIRECTORY include DESTINATION .)
install(TARGETS par EXPORT par-config)
install(EXPORT par-config NAMESPACE par:: DESTINATION lib/cmake)

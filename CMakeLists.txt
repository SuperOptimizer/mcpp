cmake_minimum_required(VERSION 3.28)
project(mcpp LANGUAGES CXX)

# ---- standard ----
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# ---- options ----
option(MCPP_BUILD_TESTS "Build mcpp tests" ON)
option(MCPP_FAST_MATH "Build with -ffast-math (expected production mode)" OFF)
option(MCPP_ASAN "Build with AddressSanitizer + UBSan" OFF)
option(MCPP_FUZZ "Build fuzz targets (Clang/libFuzzer only)" OFF)

# ---- the library: header-only foundations for now ----
add_library(mcpp INTERFACE)
add_library(mcpp::mcpp ALIAS mcpp)
target_include_directories(mcpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(mcpp INTERFACE cxx_std_26)

# Warnings are applied to consumers we build (tests), not forced on the interface.
add_library(mcpp_warnings INTERFACE)
target_compile_options(mcpp_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(MCPP_FAST_MATH)
  # Project decision: -ffast-math is an expected production build mode.
  # NaN/Inf detection therefore goes through mcpp::fpbits bit-twiddling, never std::isnan.
  target_compile_options(mcpp INTERFACE -ffast-math)
endif()

if(MCPP_ASAN)
  add_library(mcpp_sanitize INTERFACE)
  target_compile_options(mcpp_sanitize INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(mcpp_sanitize INTERFACE -fsanitize=address,undefined)
endif()

if(MCPP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(MCPP_FUZZ)
  add_subdirectory(fuzz)
endif()

# Fuzz targets. Enabled with -DMCPP_FUZZ=ON (Clang only — uses -fsanitize=fuzzer).
# Each target is a libFuzzer binary; AFL++ can consume the same
# LLVMFuzzerTestOneInput entry via afl-clang-fast if desired.

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR "MCPP_FUZZ requires Clang (libFuzzer).")
endif()

function(mcpp_add_fuzzer name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE mcpp::mcpp)
  target_compile_options(${name} PRIVATE -g -O1 -fsanitize=fuzzer,address,undefined)
  target_link_options(${name} PRIVATE -fsanitize=fuzzer,address,undefined)
endfunction()

mcpp_add_fuzzer(fuzz_block_decode)
mcpp_add_fuzzer(fuzz_chunk_view)
mcpp_add_fuzzer(fuzz_roundtrip)

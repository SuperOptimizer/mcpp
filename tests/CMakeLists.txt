# Each test is a standalone executable using mcpp's own test framework,
# registered with ctest. Built against both Clang and GCC via presets.

function(mcpp_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE mcpp::mcpp mcpp_warnings)
  if(TARGET mcpp_sanitize)
    target_link_libraries(${name} PRIVATE mcpp_sanitize)
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

mcpp_add_test(test_fpbits)
mcpp_add_test(test_mdview)
mcpp_add_test(test_dtype)
mcpp_add_test(test_toleration)  # exercises the test framework's own comparators
mcpp_add_test(test_dct)
mcpp_add_test(test_quant)
mcpp_add_test(test_range_coder)
mcpp_add_test(test_block)
mcpp_add_test(test_codec)
mcpp_add_test(test_air)

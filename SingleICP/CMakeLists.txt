include(spsArgs)

set(TARGET_NAME SingleICPMain)
add_executable(${TARGET_NAME}
    main.cpp
)

target_link_libraries(${TARGET_NAME} PRIVATE
  ICP
)

sps_mkl_verify_no_mixed_openmp_ldd(${TARGET_NAME})

# Create symlink and install shell completion
sps_args_completion(${TARGET_NAME})

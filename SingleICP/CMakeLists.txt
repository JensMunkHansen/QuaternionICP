include(spsArgs)

set(TARGET_NAME SingleICPMain)
add_executable(${TARGET_NAME}
    main.cpp
)

target_link_libraries(${TARGET_NAME} PRIVATE
  ICP
)

if(USE_MKL)
  sps_mkl_verify_no_mixed_openmp_ldd(${TARGET_NAME})
endif()

# Create symlink and install shell completion
sps_args_completion(${TARGET_NAME})

if (BUILD_SHARED_LIBS)
  # Copy DLLs for shared library builds on Windows
  sps_copy_runtime_dlls(${TARGET_NAME})
endif()

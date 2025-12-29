include(spsArgs)

add_executable(MultiICPMain
    main.cpp
)

target_link_libraries(MultiICPMain PRIVATE
    ICP
)

# Create symlink and install shell completion
sps_args_completion(MultiICPMain)

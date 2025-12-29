include(spsArgs)

add_executable(SingleICPMain
    main.cpp
)

target_link_libraries(SingleICPMain PRIVATE
    ICP
)

# Create symlink and install shell completion
sps_args_completion(SingleICPMain)

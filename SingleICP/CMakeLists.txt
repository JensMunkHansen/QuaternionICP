include(spsArgs)

add_executable(SingleICPMain
    main.cpp
)

target_link_libraries(SingleICPMain PRIVATE
    ICP
)

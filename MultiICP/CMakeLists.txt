include(spsArgs)

add_executable(MultiICPMain
    main.cpp
)

target_link_libraries(MultiICPMain PRIVATE
    ICP
)

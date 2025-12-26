add_executable(TestCorrespondences
    TestCorrespondences.cpp
)

target_link_libraries(TestCorrespondences PRIVATE
    ICP
    Catch2::Catch2WithMain
)

catch_discover_tests(TestCorrespondences)

add_executable(TestSE3
    TestSE3.cpp
)

target_link_libraries(TestSE3 PRIVATE
    ICP
    Catch2::Catch2WithMain
)

catch_discover_tests(TestSE3)

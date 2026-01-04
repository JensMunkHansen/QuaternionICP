# Common labels to propagate from Catch2 tags to ctest labels
set(ICP_TEST_LABELS python cuda backends gridsearch embree)

macro(add_icp_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ICP Catch2::Catch2WithMain)
    sps_catch_discover_tests(${name} LABELS ${ICP_TEST_LABELS})
    sps_copy_runtime_dlls(${name})
endmacro()

add_icp_test(AllBackendsTest)
add_icp_test(CorrespondencesTest)
add_icp_test(SE3Test)
add_icp_test(JacobiansAmbientTest)
add_icp_test(JacobiansAmbientTwoPoseTest)
add_icp_test(ICPSimpleTest)
add_icp_test(ICPCeresTest)
add_icp_test(ICPCeresTwoPoseTest)
add_icp_test(ICPLevenbergMarquardtTest)
add_icp_test(ParameterPropagationTest)

# Embree-specific tests
if(USE_EMBREE)
    add_icp_test(EmbreeTest)
endif()

# Backend comparison requires both backends
if(USE_EMBREE AND USE_GRIDSEARCH)
    add_icp_test(BackendComparisonTest)
endif()

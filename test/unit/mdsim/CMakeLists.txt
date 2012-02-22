add_subdirectory(integrators)
add_subdirectory(positions)
add_subdirectory(velocities)

# module box
if (WITH_CUDA)
  add_executable(test_unit_mdsim_box
    box.cpp
    box_kernel.cu
)
else (WITH_CUDA)
  add_executable(test_unit_mdsim_box
    box.cpp
)
endif (WITH_CUDA)
target_link_libraries(test_unit_mdsim_box
  libhalmd
  ${HALMD_COMMON_LIBRARIES}
  ${HALMD_TEST_LIBRARIES}
)
add_test(unit/mdsim/box/construction/2d
  test_unit_mdsim_box --run_test=box_construction_2d --log_level=test_suite
)
add_test(unit/mdsim/box/construction/3d
  test_unit_mdsim_box --run_test=box_construction_3d --log_level=test_suite
)
add_test(unit/mdsim/box/periodic/host/2d
  test_unit_mdsim_box --run_test=box_periodic_host_2d --log_level=test_suite
)
add_test(unit/mdsim/box/periodic/host/3d
  test_unit_mdsim_box --run_test=box_periodic_host_3d --log_level=test_suite
)
if(WITH_CUDA)
  add_test(unit/mdsim/box/periodic/gpu/2d
    test_unit_mdsim_box --run_test=box_periodic_gpu_2d --log_level=test_suite
  )
  add_test(unit/mdsim/box/periodic/gpu/3d
    test_unit_mdsim_box --run_test=box_periodic_gpu_3d --log_level=test_suite
  )
endif(WITH_CUDA)

# module clock
add_executable(test_unit_mdsim_clock
  clock.cpp
)
target_link_libraries(test_unit_mdsim_clock
  libhalmd
  ${HALMD_COMMON_LIBRARIES}
  ${HALMD_TEST_LIBRARIES}
)
add_test(unit/mdsim/clock
  test_unit_mdsim_clock --log_level=test_suite
)
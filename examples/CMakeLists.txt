# FZModules Examples CMakeLists.txt

#######################
### EXAMPLE PROGRAMS ###
#######################

# Demo example executable
add_executable(fzmod_demo demo.cc)
set_source_files_properties(demo.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(fzmod_demo 
    PRIVATE 
    fzmod_compile_settings
    fz_utils
    fzmod_cu_fzg
    fzmod_cu_histogram
    fzmod_cu_huffman
    fzmod_cu_lrz
    fzmod_cu_spline
    fzmod
    CUDA::cudart
)

# Set output directory for examples during build
set_target_properties(fzmod_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
)

####################
### INSTALLATION ###
####################

# Install examples to examples subdirectory
install(TARGETS fzmod_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
    COMPONENT examples
)

# Optionally install example source files for reference
install(FILES demo.cc
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/fzmod/examples
    COMPONENT examples
)

# Add a custom target to build all examples
add_custom_target(examples 
    DEPENDS fzmod_demo
    COMMENT "Building all FZModules examples"
)

# Helper function for adding new examples (for future use)
function(add_fzmod_example example_name source_file)
    add_executable(${example_name} ${source_file})
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${example_name}
        PRIVATE 
        fzmod_compile_settings
        fz_utils
        fzmod_cu_fzg
        fzmod_cu_histogram
        fzmod_cu_huffman
        fzmod_cu_lrz
        fzmod_cu_spline
        fzmod
        CUDA::cudart
    )
    set_target_properties(${example_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )
    install(TARGETS ${example_name}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
        COMPONENT examples
    )
    add_dependencies(examples ${example_name})
endfunction()

# Example usage of the helper function (uncomment when adding new examples):
# add_fzmod_example(my_new_example my_new_example.cc)
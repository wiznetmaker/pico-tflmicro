#aux_source_directory(. DIR_LIB_SRCS)
#add_library(arducam ${DIR_LIB_SRCS})
add_library(arducam
   arducam.c
   arducam.h
   ov2640.h
 )
 
 target_include_directories(arducam
   PUBLIC
   ${CMAKE_CURRENT_LIST_DIR}/.
 )
 
 target_link_libraries(arducam pico_stdlib hardware_i2c hardware_spi)
# enable usb output, disable uart output

# if (NOT TARGET arducam)
#     add_library(arducam INTERFACE)
#     target_sources(arducam INTERFACE
#             ${CMAKE_CURRENT_LIST_DIR}/arducam.c 
#     )
#     target_link_libraries(arducam INTERFACE pico_stdlib hardware_i2c hardware_spi hardware_irq)
# endif()
   
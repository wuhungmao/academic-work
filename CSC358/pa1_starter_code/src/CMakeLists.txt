file(GLOB LIB_SOURCES "*.cc")

add_library(compNet_debug STATIC ${LIB_SOURCES})

add_library(compNet_sanitized EXCLUDE_FROM_ALL STATIC ${LIB_SOURCES})
target_compile_options(compNet_sanitized PUBLIC ${SANITIZING_FLAGS})

add_library(compNet_optimized EXCLUDE_FROM_ALL STATIC ${LIB_SOURCES})
target_compile_options(compNet_optimized PUBLIC "-O2")

macro(add_app exec_name)
  add_executable("${exec_name}" "${exec_name}.cc")
  target_link_libraries("${exec_name}" compNet_debug)
  target_link_libraries("${exec_name}" util_debug)
endmacro(add_app)

add_app(webget)

if(NOT PKG_CONFIG_FOUND)
    INCLUDE(FindPkgConfig)
endif()
PKG_CHECK_MODULES(PC_WAVELEARNER wavelearner)

FIND_PATH(
    WAVELEARNER_INCLUDE_DIRS
    NAMES wavelearner/api.h
    HINTS $ENV{WAVELEARNER_DIR}/include
        ${PC_WAVELEARNER_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    WAVELEARNER_LIBRARIES
    NAMES gnuradio-wavelearner
    HINTS $ENV{WAVELEARNER_DIR}/lib
        ${PC_WAVELEARNER_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/wavelearnerTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(WAVELEARNER DEFAULT_MSG WAVELEARNER_LIBRARIES WAVELEARNER_INCLUDE_DIRS)
MARK_AS_ADVANCED(WAVELEARNER_LIBRARIES WAVELEARNER_INCLUDE_DIRS)

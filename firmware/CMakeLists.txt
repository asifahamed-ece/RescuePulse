cmake_minimum_required(VERSION 3.16.0)

# Project-level CMakeLists for the standalone ESP-IDF build.
# (Use `idf.py -p /path/to/build -S . build` from the firmware/ directory,
#  or build the PlatformIO variant in ../Rescue_Pulse_PIO instead.)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(RescuePulse)

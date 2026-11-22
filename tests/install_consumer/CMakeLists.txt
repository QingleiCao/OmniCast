# A C program built against an installed OmniCast (find_package), as an
# external project would use it. CI builds and runs it after `cmake --install`.
cmake_minimum_required(VERSION 3.18)
project(omnicast_consumer C)
find_package(OmniCast 2.0 REQUIRED)
add_executable(omnicast_consumer main.c)
target_link_libraries(omnicast_consumer PRIVATE OmniCast::omnicast)

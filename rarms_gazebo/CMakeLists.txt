CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(rarms_gazebo)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
## Compile as C++17, supported in ROS Noetic

include (FindPkgConfig)

if (PKG_CONFIG_FOUND)
    pkg_check_modules(GAZEBO gazebo)
endif()

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    message_generation
    geometry_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES contact_republisher
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
    CATKIN_DEPENDS gazebo_msgs message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

add_executable(gazebo_contacts src/contacts.cpp)

target_link_libraries(gazebo_contacts
   ${catkin_LIBRARIES}
)

target_link_libraries(gazebo_contacts ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} pthread)



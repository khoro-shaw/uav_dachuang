#!/bin/bash
source ~/.bashrc
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 5; roslaunch px4 custom_mavros_posix_sitl.launch; exec bash"' \
--tab -e 'bash -c "sleep 10; rosrun offboard_py offbn_node.py; exec bash"' \

# ~/PX4-Autopilot/launch
# ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds
# # ------------------------------------
# source ~/catkin_ws/devel/setup.bash
# source ~/uav_ws/devel/setup.bash 
# source /home/roller/catkin_ws/devel/setup.bash
# source /home/roller/uav_ws/devel/setup.bash 
# source ~/PX4-Autopilot/Tools/simulation/gazebo-classic/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default

# #---------------------------
# export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot
# export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic

# #-----------gazebo plugin------------
# export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:~/gazebo_plugin_tutorial/build
# export GAZEBO_PLUGIN_PATH=$HOME/gazebo_plugin_tutorial/build:$GAZEBO_PLUGIN_PATH
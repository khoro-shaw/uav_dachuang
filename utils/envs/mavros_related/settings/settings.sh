#!/bin/bash
source ~/.bashrc
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 5; roslaunch px4 custom_mavros_posix_sitl.launch; exec bash"' \
--tab -e 'bash -c "sleep 10; rosrun offboard_py offbn_node.py; exec bash"' \

# ~/PX4-Autopilot/launch
# ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds
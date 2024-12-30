#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import (
    CommandBool,
    CommandBoolRequest,
    SetMode,
    SetModeRequest,
)
import numpy as np

current_state = State()


def state_cb(msg):
    global current_state
    current_state = msg


def pos_cb(msg):
    global current_pos
    current_pos = msg


def vel_cb(msg):
    global current_vel
    current_vel = msg


if __name__ == "__main__":
    rospy.init_node("offb_node_py")

    # Subscribe (get information)------------------------------

    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)

    local_pos_sub = rospy.Subscriber(
        "mavros/local_position/pose", PoseStamped, callback=pos_cb
    )

    local_vel_sub = rospy.Subscriber(
        "mavros/local_position/velocity", TwistStamped, callback=vel_cb
    )

    # Publish (give information)------------------------------

    local_pos_pub = rospy.Publisher(
        "mavros/setpoint_position/local", PoseStamped, queue_size=10
    )

    local_vel_pub = rospy.Publisher(
        "mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10
    )

    # Setpoint publishing MUST be faster than 2Hz
    herz = 20.0
    rate = rospy.Rate(herz)

    # Service (interaction)--------------------------

    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
    arming_request = CommandBoolRequest()  # CommandBool这个srv的Request
    arming_request.value = True

    rospy.wait_for_service("/mavros/set_mode")
    setmode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    setmode_request = SetModeRequest()  # SetMode这个srv的Request

    # show time!!!!------------------------------

    # Wait for Flight Controller connection
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()

    testing_position = PoseStamped()
    testing_velocity = Twist()

    t0 = 0.0
    last_requset_time = rospy.Time.now()
    while not rospy.is_shutdown():

        if current_state.mode != "OFFBOARD" and (
            rospy.Time.now() - last_requset_time
        ) > rospy.Duration(5.0):
            setmode_request.custom_mode = "OFFBOARD"
            if setmode_client.call(setmode_request).mode_sent == True:
                rospy.loginfo("OFFBOARD enabled")
            last_requset_time = rospy.Time.now()

        elif not current_state.armed and (
            rospy.Time.now() - last_requset_time
        ) > rospy.Duration(5.0):
            if arming_client.call(arming_request).success == True:
                rospy.loginfo("Vehicle armed")
            last_requset_time = rospy.Time.now()

        elif (rospy.Time.now() - last_requset_time) > rospy.Duration(5.0):
            t0 += 1.0 / herz
            if current_pos.pose.position.z < 4.5:
                testing_position.pose.position.x = 0.0
                testing_position.pose.position.y = 0.0
                testing_position.pose.position.z = 5.0
                # testing_position.pose.orientation.x = 0.0
                # testing_position.pose.orientation.y = 0.0
                # testing_position.pose.orientation.z = np.sin(0.5 * np.pi * t0)
                # testing_position.pose.orientation.w = np.cos(0.5 * np.pi * t0)

            if current_pos.pose.position.z >= 4.5:
                testing_velocity.linear.x = 1.0
                testing_velocity.linear.y = 1.0
            else:
                testing_velocity.linear.x = 0.0
                testing_velocity.linear.y = 0.0
            testing_velocity.linear.z = 0.0
            testing_velocity.angular.x = 0.0
            testing_velocity.angular.y = 0.0
            testing_velocity.angular.z = 0.0

        local_vel_pub.publish(testing_velocity)
        if current_pos.pose.position.z < 4.5:
            local_pos_pub.publish(testing_position)

        rate.sleep()

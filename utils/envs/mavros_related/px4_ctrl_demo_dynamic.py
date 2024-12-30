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
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, SetModelStateRequest

import numpy as np
import torch


"""
    算法与MAVROS环境的接口
    与mavros中相关的msg和srv文件的用法，参考
    https://wiki.ros.org/mavros
    https://wiki.ros.org/mavros_extras
    这两个官方网址里面提供的解析（写得很难让人读懂）
    可以部分参考
    https://blog.csdn.net/z1872385/article/details/124606883
    https://blog.csdn.net/qq_35598561/article/details/131284168
    或者参考本人整理的rospy_related.xmind文件
"""


class EnvMavrosTest:
    def __init__(
        self,
        seed_range=150,
        privileged=False,
    ):
        self.seed_range = seed_range
        self.privileged = privileged
        self.position_input = [0.0 for i in range(7)]
        self.velocity_input = [0.0 for i in range(6)]
        self.position_output = None
        self.velocity_output = None
        self.current_state = None
        self.current_pos = None
        self.current_vel = None

        self.init_node()

    def init_node(self):
        # # ---------------ROS1 stuff--------------------------
        rospy.init_node("offb_node_py")

        def state_cb(msg):
            self.current_state = msg

        def pos_cb(msg):
            self.current_pos = msg

        def vel_cb(msg):
            self.current_vel = msg

        self.current_state = State()

        # Subscribe (get information)------------------------------

        self.state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)

        self.local_pos_sub = rospy.Subscriber(
            "mavros/local_position/pose", PoseStamped, callback=pos_cb
        )

        self.local_vel_sub = rospy.Subscriber(
            "mavros/local_position/velocity_body", TwistStamped, callback=vel_cb
        )

        # Publish (give information)------------------------------

        self.local_pos_pub = rospy.Publisher(
            "mavros/setpoint_position/local", PoseStamped, queue_size=10
        )

        self.local_vel_pub = rospy.Publisher(
            "mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10
        )

        # Setpoint publishing MUST be faster than 2Hz
        period = 0.1  # seconds
        self.rate = rospy.Rate(1.0 / period)

        # Service (interaction)--------------------------

        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.arming_request = CommandBoolRequest()  # CommandBool这个srv的Request
        self.arming_request.value = True

        rospy.wait_for_service("/mavros/set_mode")
        self.setmode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.setmode_request = SetModeRequest()  # SetMode这个srv的Request

        rospy.wait_for_service("/gazebo/set_model_state")
        self.setmodelstate_client = rospy.ServiceProxy(
            "gazebo/set_model_state", SetModelState
        )
        self.setmodelstate_request = (
            SetModelStateRequest()
        )  # SetModelState这个srv的Request

        # show time!!!!------------------------------

        # Wait for Flight Controller connection
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()

        self.testing_position = PoseStamped()
        self.testing_velocity = Twist()

    def step(self, action):
        self.position_input = action[0]
        self.velocity_input = action[1]

        self.testing_position.pose.position.x = self.position_input[0]
        self.testing_position.pose.position.y = self.position_input[1]
        self.testing_position.pose.position.z = self.position_input[2]
        self.testing_position.pose.orientation.x = self.position_input[3]
        self.testing_position.pose.orientation.y = self.position_input[4]
        self.testing_position.pose.orientation.z = self.position_input[5]
        self.testing_position.pose.orientation.w = self.position_input[6]

        self.testing_velocity.linear.x = self.velocity_input[0]
        self.testing_velocity.linear.y = self.velocity_input[1]
        self.testing_velocity.linear.z = self.velocity_input[2]
        self.testing_velocity.angular.x = self.velocity_input[3]
        self.testing_velocity.angular.y = self.velocity_input[4]
        self.testing_velocity.angular.z = self.velocity_input[5]

        if (not self.current_state.armed) or self.current_state.mode != "OFFBOARD":
            last_requset_time = rospy.Time.now()
            while (
                not self.current_state.armed
            ) or self.current_state.mode != "OFFBOARD":

                if self.current_state.mode != "OFFBOARD" and (
                    rospy.Time.now() - last_requset_time
                ) > rospy.Duration(5.0):
                    self.setmode_request.custom_mode = "OFFBOARD"
                    if self.setmode_client.call(self.setmode_request).mode_sent == True:
                        rospy.loginfo("OFFBOARD enabled")
                    last_requset_time = rospy.Time.now()

                elif not self.current_state.armed and (
                    rospy.Time.now() - last_requset_time
                ) > rospy.Duration(5.0):
                    if self.arming_client.call(self.arming_request).success == True:
                        rospy.loginfo("Vehicle armed")
                    last_requset_time = rospy.Time.now()

                self.local_vel_pub.publish(self.testing_velocity)
                self.local_pos_pub.publish(self.testing_position)

                self.rate.sleep()

        else:
            self.local_vel_pub.publish(self.testing_velocity)
            self.local_pos_pub.publish(self.testing_position)
            # rospy.loginfo(f"{i}th reply")
            # rospy.loginfo(
            #     f"{self.current_pos.pose.position.x}, {self.current_pos.pose.position.y}, {self.current_pos.pose.position.z}"
            # )
            # rospy.loginfo(
            #     f"{self.current_pos.pose.orientation.x}, {self.current_pos.pose.orientation.y}, {self.current_pos.pose.orientation.z}, {self.current_pos.pose.orientation.w}"
            # )
            # rospy.loginfo(
            #     f"{self.current_vel.twist.linear.x}, {self.current_vel.twist.linear.y}, {self.current_vel.twist.linear.z}"
            # )
            # rospy.loginfo(
            #     f"{self.current_vel.twist.angular.x}, {self.current_vel.twist.angular.y}, {self.current_vel.twist.angular.z}"
            # )
        return (
            self.current_pos.pose.position.x,
            self.current_pos.pose.position.y,
            self.current_pos.pose.position.z,
            self.current_pos.pose.orientation.x,
            self.current_pos.pose.orientation.y,
            self.current_pos.pose.orientation.z,
            self.current_pos.pose.orientation.w,
            self.current_vel.twist.linear.x,
            self.current_vel.twist.linear.y,
            self.current_vel.twist.linear.z,
            self.current_vel.twist.angular.x,
            self.current_vel.twist.angular.y,
            self.current_vel.twist.angular.z,
        )

    def reset(self):
        self.setmodelstate_request.model_state.model_name = "iris_depth_camera"
        self.setmodelstate_request.model_state.pose.position.x = 0.0
        self.setmodelstate_request.model_state.pose.position.y = 0.0
        self.setmodelstate_request.model_state.pose.position.z = 0.0
        self.setmodelstate_request.model_state.pose.orientation.x = 0.0
        self.setmodelstate_request.model_state.pose.orientation.y = 0.0
        self.setmodelstate_request.model_state.pose.orientation.z = 0.0
        self.setmodelstate_request.model_state.pose.orientation.w = 0.0
        self.setmodelstate_request.model_state.twist.linear.x = 0.0
        self.setmodelstate_request.model_state.twist.linear.y = 0.0
        self.setmodelstate_request.model_state.twist.linear.z = 0.0
        self.setmodelstate_request.model_state.twist.angular.x = 0.0
        self.setmodelstate_request.model_state.twist.angular.y = 0.0
        self.setmodelstate_request.model_state.twist.angular.z = 0.0

        while 1:
            if (
                self.setmodelstate_client.call(self.setmodelstate_request).success
                == True
            ):
                rospy.loginfo("env resetted")
                break

        return (
            self.current_pos.pose.position.x,
            self.current_pos.pose.position.y,
            self.current_pos.pose.position.z,
            self.current_pos.pose.orientation.x,
            self.current_pos.pose.orientation.y,
            self.current_pos.pose.orientation.z,
            self.current_pos.pose.orientation.w,
            self.current_vel.twist.linear.x,
            self.current_vel.twist.linear.y,
            self.current_vel.twist.linear.z,
            self.current_vel.twist.angular.x,
            self.current_vel.twist.angular.y,
            self.current_vel.twist.angular.z,
        )


test_model = EnvMavrosTest()

action_list = []
action0 = torch.zeros(size=(1, 7)).squeeze(dim=0).tolist()  # pos
action1 = torch.zeros(size=(1, 6)).squeeze(dim=0).tolist()  # vel
action_list.append(action0)
action_list.append(action1)

test_model.reset()

i = 0
while not rospy.is_shutdown():
    # --------------------------------
    action_list[0][2] = 5.0
    action_list[0][-1] = torch.cos(torch.tensor(torch.pi * i / 20.0)).item()
    action_list[0][-2] = torch.sin(torch.tensor(torch.pi * i / 20.0)).item()
    # --------------以上可用神经网络取代-------------------------

    # rospy.loginfo(f"{i}th")
    # 得到的值可以再次喂给神经网络
    test_model.step(action=action_list)

    if i == 100 - 1:
        test_model.reset()
        i = -1

    i += 1
    test_model.rate.sleep()

#!/usr/bin/env python3


import rospy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import (
    CommandBool,
    CommandBoolRequest,
    CommandTOL,
    CommandTOLRequest,
    SetMode,
    SetModeRequest,
)

from std_srvs.srv import Empty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    算法与MAVROS环境的接口
    与mavros中相关的msg和srv文件的用法，参考
    https://wiki.ros.org/mavros
    https://wiki.ros.org/mavros_extras
    这两个官方网址里面提供的解析（写得很难让人读懂）
    可以部分参考
    https://blog.csdn.net/z1872385/article/details/124606883
    https://blog.csdn.net/qq_35598561/article/details/131284168
    https://blog.csdn.net/q_12_qede/article/details/138182077
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
        self.uav_model_name = "iris_depth_camera"

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

        self.state_sub = rospy.Subscriber(
            "mavros/state", State, callback=state_cb, queue_size=1
        )

        self.local_pos_sub = rospy.Subscriber(
            "mavros/local_position/pose", PoseStamped, callback=pos_cb, queue_size=1
        )

        self.local_vel_sub = rospy.Subscriber(
            "mavros/local_position/velocity_body",
            TwistStamped,
            callback=vel_cb,
            queue_size=1,
        )

        # Publish (give information)------------------------------

        self.local_pos_pub = rospy.Publisher(
            "mavros/setpoint_position/local", PoseStamped, queue_size=1
        )

        self.local_vel_pub = rospy.Publisher(
            "mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=1
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

        rospy.wait_for_service("/mavros/cmd/land")
        self.land_client = rospy.ServiceProxy("mavros/cmd/land", CommandTOL)
        self.land_request = CommandTOLRequest()  # CommandTOL这个srv的Request

        rospy.wait_for_service("/mavros/cmd/land")
        self.land_client = rospy.ServiceProxy("mavros/cmd/land", CommandTOL)
        self.land_request = CommandTOLRequest()  # CommandTOL这个srv的Request

        rospy.wait_for_service("/gazebo/reset_world")
        self.resetworld_client = rospy.ServiceProxy("gazebo/reset_world", Empty)

        # show time!!!!------------------------------

        # Wait for Flight Controller connection
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()

        self.testing_position = PoseStamped()
        self.testing_velocity = Twist()

        self.setmode_request.custom_mode = "OFFBOARD"
        self.arming_request.value = True
        self.clear()

        if not self.current_state.armed:
            last_requset_time = rospy.Time.now()

            while not self.current_state.armed:

                if not self.current_state.armed and (
                    rospy.Time.now() - last_requset_time
                ) > rospy.Duration(5.0):
                    if self.arming_client.call(self.arming_request).success == True:
                        rospy.loginfo("Vehicle armed")
                    last_requset_time = rospy.Time.now()

                self.local_vel_pub.publish(self.testing_velocity)
                self.local_pos_pub.publish(self.testing_position)

                self.rate.sleep()

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

        self.local_vel_pub.publish(self.testing_velocity)
        self.local_pos_pub.publish(self.testing_position)

        state_array = np.array(
            [
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
            ]
        )

        return state_array

    def reset(self):
        rospy.loginfo("Env Re-setting")
        self.clear()

        self.setmode_request.custom_mode = "AUTO.RTL"
        self.arming_request.value = True

        if self.current_state.mode != "AUTO.RTL":
            last_requset_time = rospy.Time.now()

            while self.current_state.mode != "AUTO.RTL":
                if self.current_state.mode != "AUTO.RTL" and (
                    rospy.Time.now() - last_requset_time
                ) > rospy.Duration(5.0):
                    if self.setmode_client.call(self.setmode_request).mode_sent == True:
                        rospy.loginfo("AUTO.RTL enabled")
                    last_requset_time = rospy.Time.now()

                self.local_vel_pub.publish(self.testing_velocity)
                self.local_pos_pub.publish(self.testing_position)

                self.rate.sleep()

        # self.resetworld_client()

        self.setmode_request.custom_mode = "OFFBOARD"
        self.arming_request.value = True

        if (not self.current_state.armed) or self.current_state.mode != "OFFBOARD":
            last_requset_time = rospy.Time.now()
            while (
                not self.current_state.armed
            ) or self.current_state.mode != "OFFBOARD":
                if self.current_state.mode != "OFFBOARD" and (
                    rospy.Time.now() - last_requset_time
                ) > rospy.Duration(5.0):
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

        state_array = np.array(
            [
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
            ]
        )
        rospy.loginfo("Env Re-set")

        return state_array

    def clear(self):
        self.testing_position.pose.position.x = 0.0
        self.testing_position.pose.position.y = 0.0
        self.testing_position.pose.position.z = 0.0
        self.testing_position.pose.orientation.x = 0.0
        self.testing_position.pose.orientation.y = 0.0
        self.testing_position.pose.orientation.z = 0.0
        self.testing_position.pose.orientation.w = 0.0

        self.testing_velocity.linear.x = 0.0
        self.testing_velocity.linear.y = 0.0
        self.testing_velocity.linear.z = 0.0
        self.testing_velocity.angular.x = 0.0
        self.testing_velocity.angular.y = 0.0
        self.testing_velocity.angular.z = 0.0


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(in_features=13, out_features=26)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=26, out_features=13)
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x


test_model = EnvMavrosTest()

action_list = []
action0 = torch.zeros(size=(1, 7)).squeeze(dim=0).tolist()  # pos
action1 = torch.zeros(size=(1, 6)).squeeze(dim=0).tolist()  # vel
action_list.append(action0)
action_list.append(action1)

state_array = test_model.reset()


for j in range(3):
    rospy.loginfo(f"-------{j}th, iteration----------")
    for i in range(100):
        # --------------------------------
        action_list[0][0] = 3.0
        action_list[0][1] = 4.0
        action_list[0][2] = 5.0
        action_list[0][-2] = torch.sin(torch.tensor(torch.pi * i / 20.0)).item()
        action_list[0][-1] = torch.cos(torch.tensor(torch.pi * i / 20.0)).item()

        # --------------以上可用神经网络取代-------------------------

        # 得到的值可以再次喂给神经网络
        state_array = test_model.step(action=action_list)

        test_model.rate.sleep()
    test_model.reset()

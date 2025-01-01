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

from .env_base import EnvBase


"""
    算法与MAVROS环境的接口
    与mavros中相关的msg和srv文件的用法，参考
    https://wiki.ros.org/mavros
    https://wiki.ros.org/mavros_extras
    这两个官方网址里面提供的解析（写得很难让人读懂）
    可以部分参考
    https://blog.csdn.net/z1872385/article/details/124606883
    或者参考本人整理的rospy_related.xmind文件
"""


class EnvMavrosGazebo(EnvBase):
    def __init__(
        self,
        seed_range=150,
        privileged=False,
    ):
        self.seed_range = seed_range
        self.privileged = privileged
        self.position_input = None
        self.velocity_input = None
        self.position_output = None
        self.velocity_output = None
        self.current_state
        self.current_pos
        self.current_vel

        # ---------------ROS1 stuff--------------------------
        def state_cb(msg):
            self.current_state = msg

        def pos_cb(msg):
            self.current_pos = msg

        def vel_cb(msg):
            self.current_vel = msg

        self.current_state = State()
        rospy.init_node("offb_node_py")

        # Subscribe (get information)------------------------------

        self.state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)

        self.local_pos_sub = rospy.Subscriber(
            "mavros/local_position/pose", PoseStamped, callback=pos_cb
        )

        self.local_vel_sub = rospy.Subscriber(
            "mavros/local_position/velocity", TwistStamped, callback=vel_cb
        )

        # Publish (give information)------------------------------

        self.local_pos_pub = rospy.Publisher(
            "mavros/setpoint_position/local", PoseStamped, queue_size=10
        )

        self.local_vel_pub = rospy.Publisher(
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
        while not rospy.is_shutdown() and not self.current_state.connected:
            rate.sleep()

        testing_position = PoseStamped()
        testing_velocity = Twist()

        last_requset_time = rospy.Time.now()
        while not rospy.is_shutdown():

            if self.current_state.mode != "OFFBOARD" and (
                rospy.Time.now() - last_requset_time
            ) > rospy.Duration(5.0):
                setmode_request.custom_mode = "OFFBOARD"
                if setmode_client.call(setmode_request).mode_sent == True:
                    rospy.loginfo("OFFBOARD enabled")
                last_requset_time = rospy.Time.now()

            elif not self.current_state.armed and (
                rospy.Time.now() - last_requset_time
            ) > rospy.Duration(5.0):
                if arming_client.call(arming_request).success == True:
                    rospy.loginfo("Vehicle armed")
                last_requset_time = rospy.Time.now()

            elif (rospy.Time.now() - last_requset_time) > rospy.Duration(5.0):

                testing_position.pose.position.x = self.position_input[0]
                testing_position.pose.position.y = self.position_input[1]
                testing_position.pose.position.z = self.position_input[2]
                testing_position.pose.orientation.x = self.position_input[3]
                testing_position.pose.orientation.y = self.position_input[4]
                testing_position.pose.orientation.z = self.position_input[5]
                testing_position.pose.orientation.w = self.position_input[6]

                testing_velocity.linear.x = self.velocity_input[0]
                testing_velocity.linear.y = self.velocity_input[1]
                testing_velocity.linear.z = self.velocity_input[2]
                testing_velocity.angular.x = self.velocity_input[3]
                testing_velocity.angular.y = self.velocity_input[4]
                testing_velocity.angular.z = self.velocity_input[5]

            self.local_vel_pub.publish(testing_velocity)
            self.local_pos_pub.publish(testing_position)

            rate.sleep()

    def step(self, action):
        self.position_input = action[0]
        self.velocity_input = action[1]
        # if not self.privileged:
        #     return reward, obs, done
        # else:
        #     return reward, obs, obs, done

    def reset(self):
        seed = np.random.randint(low=0, high=self.seed_range)
        state_tensor = self.env.reset(seed=seed)
        if not self.privileged:
            return state_tensor
        else:
            return state_tensor, state_tensor

    def get_dims_dict(self):
        dims_dict = {}
        dims_dict["actor_state_dim"] = self.env.observation_dim
        dims_dict["critic_state_dim"] = self.env.observation_dim
        dims_dict["action_dim"] = self.env.action_dim
        return dims_dict

    def get_range(self):
        obs_range = self.env.observation_range
        action_range = self.env.action_range
        if not self.privileged:
            return obs_range[0], obs_range[1], action_range[0], action_range[1]
        else:
            return (
                obs_range[0],
                obs_range[1],
                obs_range[0],
                obs_range[1],
                action_range[0],
                action_range[1],
            )

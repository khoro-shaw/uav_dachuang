#!/usr/bin/env python3


import rospy
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import (
    CommandBool,
    CommandBoolRequest,
    CommandTOL,
    CommandTOLRequest,
    SetMode,
    SetModeRequest,
)
from gazebo_msgs.msg import LinkStates, ModelStates  # get
from gazebo_msgs.msg import LinkState, ModelState  # give

from std_srvs.srv import Empty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


"""
    本代码是结合刘宇豪同志写的world文件
    所写的跟踪代码
    Gazebo: 红x，绿y，蓝z（右手）
    yaw的范围，-180°~+180°
"""


class MavrosEnv:
    def __init__(
        self,
        model="iris_depth_camera",
    ):
        # xyz的相对位移，相对yaw (4)
        # 三个线速度，三个角速度 (6)
        self.observation_dim = 10

        # 三个线速度，三个角速度
        self.action_dim = 6
        # action_range改为np.array
        self.action_range = np.array(
            object=[[-1, -1, -1, -10, -10, -10], [20, 20, 20, 10, 10, 10]],
            dtype=np.float32,
        )

        self.uav_model_name = model
        self.action_input = [0.0 for i in range(self.action_dim)]

        self.iter_counter = 0  # 计训练次数
        self.past_action = []  # 计过去的动作

        self.x_diff = 0.0
        self.y_diff = 0.0
        self.z_diff = 0.0
        self.yaw_diff = 0.0
        self.pitch_diff = 0.0
        self.x_lin = 0.0
        self.y_lin = 0.0
        self.z_lin = 0.0
        self.x_ang = 0.0
        self.y_ang = 0.0
        self.z_ang = 0.0
        self.roll_current = 0.0
        self.pitch_current = 0.0
        self.yaw_current = 0.0

        self.current_state = State()
        self.current_pos = PoseStamped()
        self.current_vel = TwistStamped()
        self.gaz_link_state = LinkStates()
        self.gaz_model_state = ModelStates()

        self.current_pos.pose.orientation.w = 1.0

        self.init_node()

    def step(self, action):
        self.iter_counter += 1

        self.action_input = action
        self.past_action.append(torch.tensor(action))

        self.testing_velocity.twist.linear.x = self.action_input[0]
        self.testing_velocity.twist.linear.y = self.action_input[1]
        self.testing_velocity.twist.linear.z = self.action_input[2]

        self.testing_velocity.twist.angular.x = self.action_input[-3]
        self.testing_velocity.twist.angular.y = self.action_input[-2]
        self.testing_velocity.twist.angular.z = self.action_input[-1]

        self.local_vel_pub.publish(self.testing_velocity)

        self.get_obs()

        state_array = np.array(
            [
                self.x_diff,
                self.y_diff,
                self.z_diff,
                self.yaw_diff,
                self.current_pos.pose.position.x,
                self.current_pos.pose.position.y,
                self.current_pos.pose.position.z,
                self.roll_current,
                self.pitch_current,
                self.yaw_current,
            ]
        )

        reward = self.reward_all()

        done = self.track_done()

        return state_array, reward, done

    def reset(self, seed=None):
        rospy.loginfo("Env Re-setting")
        self.clear()

        self.setmode_request.custom_mode = "AUTO.LAND"
        self.arming_request.value = True

        if self.current_state.mode != "AUTO.LAND":
            last_requset_time = rospy.Time.now()

            while self.current_state.mode != "AUTO.LAND":
                if self.current_state.mode != "AUTO.LAND" and (
                    rospy.Time.now() - last_requset_time
                ) > rospy.Duration(0.1):
                    if self.setmode_client.call(self.setmode_request).mode_sent == True:
                        rospy.loginfo("AUTO.LAND enabled")
                    last_requset_time = rospy.Time.now()

                self.local_vel_pub.publish(self.testing_velocity)

                self.rate.sleep()

        self.setmode_request.custom_mode = "OFFBOARD"
        self.arming_request.value = True

        state_array = np.array(
            [
                self.x_diff,
                self.y_diff,
                self.z_diff,
                self.yaw_diff,
                self.current_pos.pose.position.x,
                self.current_pos.pose.position.y,
                self.current_pos.pose.position.z,
                self.roll_current,
                self.pitch_current,
                self.yaw_current,
            ]
        )
        rospy.loginfo("Env Re-set")

        for i in range(20):
            self.rate.sleep()

        # 无人机复位
        for i in range(20):
            self.model_state_pub.publish(self.testing_model_state)
            self.rate.sleep()

        for i in range(20):
            self.rate.sleep()

        # 启动板外飞行模式和解锁
        if (not self.current_state.armed) or self.current_state.mode != "OFFBOARD":
            last_requset_time = rospy.Time.now()
            while (
                not self.current_state.armed
            ) or self.current_state.mode != "OFFBOARD":
                if self.current_state.mode != "OFFBOARD" and (
                    rospy.Time.now() - last_requset_time
                ) > rospy.Duration(0.1):
                    if self.setmode_client.call(self.setmode_request).mode_sent == True:
                        rospy.loginfo("OFFBOARD enabled")
                    last_requset_time = rospy.Time.now()

                elif not self.current_state.armed and (
                    rospy.Time.now() - last_requset_time
                ) > rospy.Duration(0.1):
                    if self.arming_client.call(self.arming_request).success == True:
                        rospy.loginfo("Vehicle armed")
                    last_requset_time = rospy.Time.now()

                self.local_vel_pub.publish(self.testing_velocity)

                self.rate.sleep()

        return state_array

    def init_node(self):
        # ---------------ROS1 stuff--------------------------
        rospy.init_node("offb_node_py")

        def state_cb(msg):
            self.current_state = msg

        def pos_cb(msg):
            self.current_pos = msg

        def vel_cb(msg):
            self.current_vel = msg

        def gaz_link_state_cb(msg):
            self.gaz_link_state = msg

        def gaz_model_state_cb(msg):
            self.gaz_model_state = msg

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

        self.gaz_link_state_sub = rospy.Subscriber(
            "gazebo/link_states",
            LinkStates,
            callback=gaz_link_state_cb,
            queue_size=1,
        )

        self.gaz_model_state_sub = rospy.Subscriber(
            "gazebo/model_states",
            ModelStates,
            callback=gaz_model_state_cb,
            queue_size=1,
        )

        # Publish (give information)------------------------------

        self.local_pos_pub = rospy.Publisher(
            "mavros/setpoint_position/local", PoseStamped, queue_size=1
        )

        self.local_vel_pub = rospy.Publisher(
            "/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1
        )

        self.link_state_pub = rospy.Publisher(
            "/gazebo/set_link_state", LinkState, queue_size=1
        )
        self.model_state_pub = rospy.Publisher(
            "/gazebo/set_model_state", ModelState, queue_size=1
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

        # show time!!!!------------------------------

        # Wait for Flight Controller connection
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()

        self.testing_position = PoseStamped()
        self.testing_velocity = TwistStamped()
        self.testing_link_state = LinkState()
        self.testing_model_state = ModelState()

        # 设置ModelStates和LinkStates（反正这两个结构体的值是不变的）
        self.testing_model_state.model_name = self.uav_model_name
        self.testing_model_state.pose.position.x = 0.0
        self.testing_model_state.pose.position.y = 0.0
        self.testing_model_state.pose.position.z = 0.5  # 缓冲量，防止穿模
        self.testing_model_state.pose.orientation.x = 0.0
        self.testing_model_state.pose.orientation.y = 0.0
        self.testing_model_state.pose.orientation.z = 0.0
        self.testing_model_state.pose.orientation.w = 1.0
        self.testing_model_state.twist.linear.x = 0.0
        self.testing_model_state.twist.linear.y = 0.0
        self.testing_model_state.twist.linear.z = 0.0
        self.testing_model_state.twist.angular.x = 0.0
        self.testing_model_state.twist.angular.y = 0.0
        self.testing_model_state.twist.angular.z = 0.0
        self.testing_model_state.reference_frame = "world"

        self.setmode_request.custom_mode = "OFFBOARD"
        self.arming_request.value = True

        # 初始化输入值
        self.clear()

        # 启动板外飞行模式和解锁
        if (not self.current_state.armed) or self.current_state.mode != "OFFBOARD":
            last_requset_time = rospy.Time.now()
            while (
                not self.current_state.armed
            ) or self.current_state.mode != "OFFBOARD":
                if self.current_state.mode != "OFFBOARD" and (
                    rospy.Time.now() - last_requset_time
                ) > rospy.Duration(0.1):
                    if self.setmode_client.call(self.setmode_request).mode_sent == True:
                        rospy.loginfo("OFFBOARD enabled")
                    last_requset_time = rospy.Time.now()

                elif not self.current_state.armed and (
                    rospy.Time.now() - last_requset_time
                ) > rospy.Duration(0.1):
                    if self.arming_client.call(self.arming_request).success == True:
                        rospy.loginfo("Vehicle armed")
                    last_requset_time = rospy.Time.now()

                self.local_vel_pub.publish(self.testing_velocity)

                self.rate.sleep()

        self.get_obs()

    def get_obs(self):
        link_names_list = self.gaz_link_state.name
        link_pos_list = self.gaz_link_state.pose

        model_names_list = self.gaz_model_state.name
        model_pos_list = self.gaz_model_state.pose

        if link_names_list is not None:
            # rospy.loginfo(link_names_list)
            # rospy.loginfo(model_names_list)
            head_idx = link_names_list.index("actor_walking::Head")
            man_idx = model_names_list.index("actor_walking")
            drone_idx = model_names_list.index(self.uav_model_name)

            # ------------------------------------------

            self.head_pose = link_pos_list[head_idx].position
            self.man_pose = model_pos_list[man_idx].position
            self.drone_pose = model_pos_list[drone_idx].position

            # self.head_orient = link_pos_list[head_idx].orientation
            self.man_orient = model_pos_list[man_idx].orientation
            self.drone_orient = model_pos_list[drone_idx].orientation

            # --------------------------------------
            # 精确位移差，姑且认为深度相机能给
            self.x_diff = self.head_pose.x - self.drone_pose.x
            self.y_diff = self.head_pose.y - self.drone_pose.y
            self.z_diff = self.head_pose.z - self.drone_pose.z
            # self.x_diff = self.man_pose.x - self.drone_pose.x
            # self.y_diff = self.man_pose.y - self.drone_pose.y
            # self.z_diff = (self.man_pose.z + 0.4) - self.drone_pose.z  # 头的高度

            # 当前无人机的欧拉角，由PX4内部数据计算而得
            self.roll_current, self.pitch_current, self.yaw_current = (
                self.quaternion_to_euler(
                    self.current_pos.pose.orientation.x,
                    self.current_pos.pose.orientation.y,
                    self.current_pos.pose.orientation.z,
                    self.current_pos.pose.orientation.w,
                )
            )  # radian

            # 无人机与人头的欧拉角，可计算xyz轴相差的角度
            self.yaw_diff = math.atan2(self.y_diff, self.x_diff)

            xy_length = math.hypot(self.x_diff, self.y_diff)
            self.pitch_diff = math.atan2(self.z_diff, xy_length)

            # 无人机的线速度与角速度
            self.x_lin = self.current_vel.twist.linear.x
            self.y_lin = self.current_vel.twist.linear.y
            self.z_lin = self.current_vel.twist.linear.z
            self.x_ang = self.current_vel.twist.angular.x
            self.y_ang = self.current_vel.twist.angular.y
            self.z_ang = self.current_vel.twist.angular.z

    def reward_all(self):
        # 目前没有加上速度相关的值
        rew_dis = self.reward_dis()
        rew_yaw = self.reward_yaw()
        rew_cont = self.reward_continuity()
        rew_air = self.reward_airborne()
        return rew_dis + rew_yaw + rew_cont + rew_air

    def reward_dis(self):
        # 与目标的距离，越近越好
        dis_sum = self.x_diff**2 + self.y_diff**2 + self.z_diff**2
        return -np.log(dis_sum).item()

    def reward_yaw(self):
        # 使无人机朝向人头的位置
        return -np.log((self.yaw_diff - self.yaw_current) ** 2).item()

    def reward_continuity(self):
        # 保证控制指令的连续性
        if len(self.past_action) < 2:
            return 0
        else:
            return -torch.log(
                torch.sum((self.past_action[-2] - self.past_action[-1]) ** 2)
            ).item()

    def reward_airborne(self):
        # 奖励上天的无人机
        return self.current_pos.pose.position.z

    def reward_lin(self):
        # 速度应该有个限度，但是不知道怎么设置
        return self.x_lin + self.y_lin + self.z_lin

    def reward_ang(self):
        return self.x_ang + self.y_ang + self.z_ang

    def track_done(self):
        if np.abs(np.abs(self.roll_current) - np.pi).item() < 10.0 / 180 * np.pi:
            return True
        elif np.abs(np.abs(self.pitch_current) - np.pi).item() < 10.0 / 180 * np.pi:
            return True
        elif np.abs(self.current_pos.pose.position.x) > 10.0:
            return True
        elif np.abs(self.current_pos.pose.position.y) > 10.0:
            return True
        elif np.abs(self.current_pos.pose.position.z) > 10.0:
            return True
        elif self.iter_counter > 5000:
            self.iter_counter = 0
            return True
        elif self.iter_counter <= 5000:  # for testing
            return False

    def clear(self):
        self.testing_velocity.twist.linear.x = 0.0
        self.testing_velocity.twist.linear.y = 0.0
        self.testing_velocity.twist.linear.z = 0.0
        self.testing_velocity.twist.angular.x = 0.0
        self.testing_velocity.twist.angular.y = 0.0
        self.testing_velocity.twist.angular.z = 0.0

    def quaternion_to_euler(self, x, y, z, w):
        """
        将四元数转换为欧拉角（弧度制），采用ZYX旋转顺序（即Yaw-Pitch-Roll）
        :param w, x, y, z: 四元数的四个分量，需满足归一化条件（w² + x² + y² + z² = 1）
        :return: (roll, pitch, yaw) 欧拉角（弧度）
        Coded by DeepSeek
        """
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        # 计算俯仰角（绕Y轴）
        sin_p = 2 * (w * y - z * x)
        sin_p = max(min(sin_p, 1.0), -1.0)  # 防止浮点误差导致超出[-1,1]范围
        pitch = math.asin(sin_p)

        # 计算翻滚角和偏航角（处理万向节锁情况）
        if abs(sin_p) >= 0.9999:  # 俯仰角接近±90°
            # 万向节锁时，仅保留yaw + roll的合成角度
            roll = 0.0
            yaw = math.atan2(2 * (x * y + w * z), w**2 + x**2 - y**2 - z**2)
        else:
            roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        yaw = yaw % (2 * math.pi)  # 确保在[0, 2π)
        if yaw > math.pi:
            yaw -= 2 * math.pi

        return roll, pitch, yaw

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        将欧拉角（弧度制）转换为四元数
        :param roll:  X轴翻滚角（弧度）
        :param pitch: Y轴俯仰角（弧度）
        :param yaw:   Z轴偏航角（弧度）
        :return: 四元数 [w, x, y, z]
        Coded by DeepSeek
        """
        # 计算半角的三角函数
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        # 四元数分量计算（ZYX顺序）
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr

        return [x, y, z, w]

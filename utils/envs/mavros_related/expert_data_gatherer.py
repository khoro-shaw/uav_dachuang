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
import pandas as pd
import time
import os


"""
    尝试使用模仿学习解决强化学习初始状态下，
    网络参数容易训练成nan的问题，
    利用位置信息，mavros/setpoint_position/local，
    给速度控制单元，/mavros/setpoint_velocity/cmd_vel提供专家数据
"""


class ExpertDataGatherer:
    def __init__(
        self,
        model="iris_depth_camera",
    ):
        # xyz的相对位移，相对yaw (4)
        # 三个线位置，三个角位置 (6)
        self.observation_dim = 10

        # xyz位置（3）
        # 以及与yaw值相关的两个四元数（2）
        self.action_dim = 5

        self.uav_model_name = model
        self.action_input = [0.0 for i in range(self.action_dim)]

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

        self.iter_counter = 0

        self.init_node()

    def step(self, action):
        self.iter_counter += 1

        self.action_input = action

        self.testing_position.pose.position.x = self.action_input[0]
        self.testing_position.pose.position.y = self.action_input[1]
        self.testing_position.pose.position.z = self.action_input[2]

        self.testing_position.pose.orientation.z = self.action_input[-2]
        self.testing_position.pose.orientation.w = self.action_input[-1]

        self.local_pos_pub.publish(self.testing_position)
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
        action_array = np.array(
            [
                self.current_vel.twist.linear.x,
                self.current_vel.twist.linear.y,
                self.current_vel.twist.linear.z,
                self.current_vel.twist.angular.x,
                self.current_vel.twist.angular.y,
                self.current_vel.twist.angular.z,
            ]
        )

        done = self.track_done()
        return state_array, action_array, done

    def reset(self, seed=None):
        """
        需要好好研究下该怎么真正地reset
        """
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

                self.local_pos_pub.publish(self.testing_position)
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

                self.local_pos_pub.publish(self.testing_position)
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

                self.local_pos_pub.publish(self.testing_position)
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

    def track_done(self):
        if self.iter_counter <= 5000:  # for testing
            return False
        elif self.iter_counter > 5000:
            self.iter_counter = 0
            return True
        elif np.abs(np.abs(self.roll_current) - np.pi).item() < 10.0 / 180 * np.pi:
            return True
        elif np.abs(np.abs(self.pitch_current) - np.pi).item() < 10.0 / 180 * np.pi:
            return True

    def clear(self):
        self.testing_position.pose.position.x = 0.0
        self.testing_position.pose.position.y = 0.0
        self.testing_position.pose.position.z = 0.0
        self.testing_position.pose.orientation.x = 0.0
        self.testing_position.pose.orientation.y = 0.0
        self.testing_position.pose.orientation.z = 0.0
        self.testing_position.pose.orientation.w = 1.0

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


test_model = ExpertDataGatherer()
action_list = [0.0 for i in range(test_model.action_dim)]
state_log = []
action_log = []

for j in range(1):
    done = False
    while not done:
        # --------------------------------
        action_list[0] = test_model.x_diff + test_model.current_pos.pose.position.x
        action_list[1] = test_model.y_diff + test_model.current_pos.pose.position.y
        action_list[2] = test_model.z_diff + test_model.current_pos.pose.position.z

        action_list[-2] = math.sin(0.5 * (test_model.yaw_diff))
        action_list[-1] = math.cos(0.5 * (test_model.yaw_diff))

        # --------------以上用于生成专家数据-------------------------

        state_array, action_array, done = test_model.step(action=action_list)

        state_log.append(state_array)
        action_log.append(action_array)

        test_model.rate.sleep()

    data = []
    for arr1, arr2 in zip(state_log, action_log):
        data.append(np.concatenate([arr1, arr2]))

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 自定义列名
    df.columns = [
        "x_diff",
        "y_diff",
        "z_diff",
        "yaw_diff",
        "x_current",
        "y_current",
        "z_current",
        "roll_current",
        "pitch_current",
        "yaw_current",
        "lin_vel_x",
        "lin_vel_y",
        "lin_vel_z",
        "ang_vel_x",
        "ang_vel_y",
        "ang_vel_z",
    ]

    # 写入 CSV
    t = time.localtime()

    folder = os.path.exists(f"./expert_data")
    if not folder:
        os.makedirs(f"./expert_data")

    df.to_csv(
        f"./expert_data/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}.csv",
        index=False,
    )

    state_array = test_model.reset()

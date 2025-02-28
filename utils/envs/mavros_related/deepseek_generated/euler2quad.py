import math


def euler_to_quaternion(roll, pitch, yaw):
    """
    将欧拉角（弧度制）转换为四元数
    :param roll:  X轴翻滚角（弧度）
    :param pitch: Y轴俯仰角（弧度）
    :param yaw:   Z轴偏航角（弧度）
    :return: 四元数 [w, x, y, z]
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

    return [w, x, y, z]


# 示例：将欧拉角（弧度）转换为四元数
roll = math.radians(0)  # X轴旋转30度
pitch = math.radians(0)  # Y轴旋转45度
yaw = math.radians(180)  # Z轴旋转60度

quaternion = euler_to_quaternion(roll, pitch, yaw)
print(
    f"四元数：w={quaternion[0]:.4f}, x={quaternion[1]:.4f}, y={quaternion[2]:.4f}, z={quaternion[3]:.4f}"
)

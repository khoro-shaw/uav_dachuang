import math


def quaternion_to_euler(w, x, y, z):
    """
    将四元数转换为欧拉角（弧度制），采用ZYX旋转顺序（即Yaw-Pitch-Roll）
    :param w, x, y, z: 四元数的四个分量，需满足归一化条件（w² + x² + y² + z² = 1）
    :return: (roll, pitch, yaw) 欧拉角（弧度）
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

    return roll, pitch, yaw


x = 0.507704804511099
y = 0.4684169869322088
z = 0.5259358887116989
w = 0.4961983462232953

# x = 0.0013859438882984898
# y = 0.7058600138276091
# z = 0.7080529182678567
# w = -0.02051304392813863

# x = 0.5531308577531423
# y = -0.42532513099668817
# z = -0.3801329619137034
# w = 0.6071603728922355

# x = 0.6860922868184459
# y = -0.1790046622012256
# z = -0.11286809927885542
# w = 0.6960571076022708

# x = 0.5038570352228856
# y = 0.4866804121584812
# z = 0.4658754028970328
# w = 0.5405833640181525


# 示例：将四元数转换为欧拉角（与历史对话中的示例对应）
quat = [
    0.7771459614569708,
    0.0,
    0.0,
    0.6293203910498375,
]  # 对应30° roll, 45° pitch, 60° yaw的四元数
roll_rad, pitch_rad, yaw_rad = quaternion_to_euler(*quat)

yaw_rad = yaw_rad % (2 * math.pi)  # 确保在[0, 2π)
if yaw_rad > math.pi:
    yaw_rad -= 2 * math.pi

# 转换为角度输出
roll_deg = math.degrees(roll_rad)
pitch_deg = math.degrees(pitch_rad)
yaw_deg = math.degrees(yaw_rad)
print(f"欧拉角：roll={roll_deg:.2f}°, pitch={pitch_deg:.2f}°, yaw={yaw_deg:.2f}°")

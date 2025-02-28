import math


def calculate_euler_angles(A, B):
    """
    计算点A相对于点B的三个欧拉角（Z-Y-X顺序）
    返回值单位为角度制，范围：
    yaw:   (-180°, 180°]
    pitch: [-90°, 90°]
    roll:  默认为0（需要额外信息才能计算）
    """
    # 计算相对坐标
    dx = A[0] - B[0]
    dy = A[1] - B[1]
    dz = A[2] - B[2]

    # 处理零向量情况
    if dx == 0 and dy == 0 and dz == 0:
        return (0.0, 0.0, 0.0)

    # 计算水平投影长度
    xy_length = math.hypot(dx, dy)

    # 计算偏航角（绕Z轴，X-Y平面旋转）
    yaw = math.degrees(math.atan2(dy, dx))

    # 计算俯仰角（绕Y轴，X-Z平面旋转）
    pitch = math.degrees(math.atan2(dz, xy_length))

    # 滚动角需要额外参考平面信息，默认设为0
    roll = 0.0

    return (yaw, pitch, roll)


# 示例测试
A = (1, 2, 3)
B = (0, 0, 0)
yaw, pitch, roll = calculate_euler_angles(A, B)

print(f"偏航角(Yaw): {yaw:.2f}°")
print(f"俯仰角(Pitch): {pitch:.2f}°")
print(f"滚动角(Roll): {roll:.2f}°（需额外信息计算）")

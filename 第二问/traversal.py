import numpy as np
import pandas as pd
from scipy.integrate import quad
import math
from itertools import zip_longest
from tqdm import tqdm
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

p = 0.55  # 螺距
v = 1  # 龙头的速度
time_acc = 0.1  # 时间精度
d = math.sqrt(0.275**2 + 0.15**2)  # 龙头前把手到两个顶角的距离（半径）
TOL = 1e-10

def r(theta):
    # 计算r
    r = (p * theta) / (2 * np.pi)
    return r


def arc_length(theta1, theta2):
    # 计算两个点之间的弧长
    return (0.55 * quad(lambda theta: np.sqrt(theta**2 + 1), theta2, theta1)[0]) / (
        2 * np.pi
    )


def find_theta_for_arc_length(theta, target_length, tol=TOL):
    # 用二分法寻找满足弧长要求的点
    lower = 0
    upper = theta
    while upper - lower > tol:
        mid = (lower + upper) / 2
        length = arc_length(theta, mid)

        if length < target_length:
            upper = mid
        else:
            lower = mid
    return (lower + upper) / 2


def unit_motion(theta):
    # 定义每次移动
    s = v * time_acc
    global t
    t = t + time_acc
    target_theta = find_theta_for_arc_length(theta, target_length=s, tol=TOL)
    return target_theta


def distance(theta1, theta2):
    # 计算两个点之间的距离
    r1 = p * theta1 / (2 * np.pi)
    r2 = p * theta2 / (2 * np.pi)
    cos_value = math.cos(float(theta2 - theta1))
    distance = math.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * cos_value)
    return float(distance)


def find_theta_for_distance(theta, target_distance, tol=TOL):
    # 对于龙身，利用二分法找到满足距离的点
    lower = theta
    upper = theta + np.pi
    while upper - lower > tol:
        mid = (lower + upper) / 2
        length = distance(theta, mid)
        if length < target_distance:
            lower = mid
        else:
            upper = mid
    return (lower + upper) / 2


def find_all_points(theta):
    list = []  # 用来储存这一秒各点位的theta值
    list.append(theta)  # 储存龙头前把手
    theta1 = find_theta_for_distance(theta, target_distance=2.86, tol=TOL)
    list.append(theta1)  # 储存龙头后把手

    final_theta = theta + 5 * np.pi / 2
    theta = theta1  # 初始化theta，准备开始循环
    while True:
        theta = find_theta_for_distance(theta, target_distance=1.65, tol=TOL)
        if theta < final_theta:
            list.append(theta)  # 储存满足条件的龙身后把手
        else:
            break  # 超过5/2 pi，跳出循环
    return list


def polar_to_cartesian(theta):
    # 将极坐标换算成直角坐标
    r = (p * theta) / (2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def line_equation(theta1, theta2):
    # 用直角坐标表示一条直线
    x1, y1 = polar_to_cartesian(theta1)
    x2, y2 = polar_to_cartesian(theta2)
    # print(theta1, theta2, x1, y1, x2, y2)
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    A = k
    B = -1
    C = b
    return A, B, C


def parallel_lines(theta1, theta2, d=0.15):
    # 得到平行线的参数
    A, B, C = line_equation(theta1, theta2)
    # collision_plot(A, B, C)
    C1 = C - math.sqrt(A**2 + B**2) * d
    C2 = C + math.sqrt(A**2 + B**2) * d
    return A, B, C1, C2


def calculate_new_position(x1, y1, x2, y2):
    # 计算方向向量
    dx = x2 - x1
    dy = y2 - y1

    # 归一化方向向量
    length = math.sqrt(dx**2 + dy**2)
    unit_dx = dx / length
    unit_dy = dy / length

    # 沿着方向向量前进0.275
    x3 = x2 + unit_dx * 0.275
    y3 = y2 + unit_dy * 0.275

    # 逆时针旋转90度的方向向量
    rotated_dx = -unit_dy
    rotated_dy = unit_dx

    # 沿着旋转后的方向向量前进0.15
    x_final = x3 + rotated_dx * 0.15
    y_final = y3 + rotated_dy * 0.15

    return x_final, y_final


def is_collide(A, B, C1, C2, list, d=0.001):
    theta1=list[1]
    theta2=list[0]
    x1, y1 = polar_to_cartesian(theta1)
    x2, y2 = polar_to_cartesian(theta2)
    x3, y3 = calculate_new_position(x1, y1, x2, y2)

    denominator = math.sqrt(A**2 + B**2)
    d1 = abs(A * x3 + B * y3 + C1) / denominator
    d2 = abs(A * x3 + B * y3 + C2) / denominator
    # print(d1,d2)
    return d1 < d or d2 < d, x3, y3


def show_plot(list, range):
    # 画出碰撞时候的图片
    font_path = "./SimHei.ttf"
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, polar=True)
    theta_values = list
    r_values = [r(theta) for theta in theta_values]
    # print(r_values)
    ax.plot(theta_values, r_values, "b")

    # 绘制极坐标曲线
    theta_values1 = np.linspace(0, range * np.pi, 1000)
    r_values1 = (p * theta_values1) / (2 * np.pi)
    ax.plot(theta_values1, r_values1, "--", alpha=0.3, label="r=(p*theta)/(2*pi)")

    ax.scatter(
        theta_values,
        r_values,
        color="r",
        s=10,
        zorder=5,
    )

    # ax.legend(prop=font_prop)
    ax.grid(False)
    plt.show()

# font_path = "./微软雅黑.ttf"
# prop = font_manager.FontProperties(fname=font_path)

def collision_plot(
    A, B, C1, C2=None, theta1=None, theta2=None, x3=None, y3=None, list=None
):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-10, 10, 400)
    y1 = (-A * x - C1) / B
    ax.plot(x, y1, label=f"{A}x + {B}y + {C1} = 0")

    if C2 is not None:
        y2 = (-A * x - C2) / B
        ax.plot(x, y2, label=f"{A}x + {B}y + {C2} = 0")

    if theta1 is not None and theta2 is not None:
        x1, y1 = polar_to_cartesian(theta1)
        x2, y2 = polar_to_cartesian(theta2)
        ax.plot(x1, y1, "go", label="龙身前把手")
        ax.plot(x2, y2, "go")

    if list is not None:
        theta_0 = list[0]
        theta_1 = list[1]
        x_0, y_0 = polar_to_cartesian(theta_0)
        x_1, y_1 = polar_to_cartesian(theta_1)
        ax.plot(x_0, y_0, "bo", label="龙头前把手")
        ax.plot(x_1, y_1, "bo")

    if x3 is not None and y3 is not None:
        ax.plot(x3, y3, "ro", label="碰撞点")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal", "box")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.show()


def All_You_Need_Is_Points(theta):
    list = []  # 用来储存这一秒各点位的theta值
    list.append(theta)  # 储存龙头前把手
    theta1 = find_theta_for_distance(theta, target_distance=2.86, tol=TOL)
    list.append(theta1)  # 储存龙头后把手

    theta = theta1  # 初始化theta为龙头的后把手，准备循环
    while True:
        theta = find_theta_for_distance(theta, target_distance=1.65, tol=TOL)
        list.append(theta)
        if len(list) == 224:  # 得到所有把手位置之后，退出循环
            break
    return list


def find_alpha_2(theta_prevous, theta):
    r_prevous = (0.55 * theta_prevous) / (2 * np.pi)
    r = (0.55 * theta) / (2 * np.pi)
    tan_alpha_2 = (r_prevous * math.sin(theta_prevous) - r * math.sin(theta)) / (
        r_prevous * math.cos(theta_prevous) - r * math.cos(theta)
    )
    alpha_2 = math.atan(tan_alpha_2)
    return alpha_2


def find_alpha(theta_prevous, theta):
    tan_alpha_1 = (
        math.sin(theta_prevous) + theta_prevous * math.cos(theta_prevous)
    ) / (math.cos(theta_prevous) - theta_prevous * math.sin(theta_prevous))
    alpha_1 = math.atan(tan_alpha_1)

    alpha_2 = find_alpha_2(theta_prevous, theta)

    alpha = abs(alpha_1 - alpha_2)
    if alpha > np.pi / 2:
        alpha = np.pi - alpha
    return alpha


def find_beta(theta_prevous, theta, l=1.65):
    tan_phi = (math.sin(theta) + theta * math.cos(theta)) / (
        math.cos(theta) - theta * math.sin(theta)
    )
    phi = math.atan(tan_phi)
    if phi < 0:
        phi = phi + np.pi

    phi2 = find_alpha_2(theta_prevous, theta)

    # print(theta % (2 * np.pi))
    beta = abs(phi - phi2)
    # print("phi, gamma, theta_0: ", phi, gamma)
    return beta


def velocity(v_prevous, alpha, beta):
    v = abs((math.cos(alpha) * v_prevous) / (math.cos(beta)))
    return v


def final_step(list):
    list = All_You_Need_Is_Points(list[0])
    show_plot(list, 32)

    x_list = []
    y_list = []
    for theta in list:
        x, y = polar_to_cartesian(theta)
        x_list.append(x)
        y_list.append(y)

    v_list = []
    v_list.append(1)
    v = 1
    for theta_prevous, theta in zip(list, list[1:]):
        if math.isnan(theta):
            break
        alpha = find_alpha(theta_prevous, theta)
        beta = find_beta(theta_prevous, theta, l=1.65)
        v = velocity(v, alpha, beta)
        v_list.append(v)

    df = pd.DataFrame({"x": x_list, "y": y_list, "v": v_list})
    df = df.astype(float)
    # 控制输出精度为小数点后六位
    df = df.round(6)
    df.to_excel("final_2.xlsx", index=False, header=False)


if __name__ == "__main__":
    theta = 32 * np.pi
    t=0
    while True:
        IS_COLLIDE = False  # 判断是否碰撞的指标
        theta = unit_motion(theta)  # 进行移动，距离是 v*time_acc
        # print(theta)
        list = find_all_points(theta)  # 得到最近一圈各点位的theta值
        # print(list)
        # show_plot(list)
        theta0 = list[0]  # 把龙头的位置单独拿出来变为圆心
        # print(theta0)

        for theta1, theta2 in zip(list[1:], list[2:]):
            # print(theta1, theta2)
            A, B, C1, C2 = parallel_lines(theta1, theta2, d=0.15)
            # collision_plot(A, B, C1, C2)
            IS_COLLIDE, x3, y3 = is_collide(A, B, C1, C2, list)
            if IS_COLLIDE:  # 如果碰撞，立即跳出循环
                break
            if theta2 is None:
                break
        if IS_COLLIDE:  # 如果碰撞，立即跳出循环
            break
        if theta < 0:
            break
        # print(theta0)
    print(f"发生碰撞! 碰撞时间为: {t} 秒")
    print(f"碰撞坐标为: x={x3} y={y3}")
    # print(d)

    # show_plot(list, 15)
    collision_plot(A, B, C1, C2, theta1, theta2, x3, y3, list)

    final_step(list)

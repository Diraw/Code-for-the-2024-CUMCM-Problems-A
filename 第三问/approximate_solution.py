import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def circle_1():
    # 定义圆参数
    r_circle = 4.5

    # 计算线段的两端点
    line_length = 2.86
    theta_diff = line_length / r_circle  # 角度差

    # 选择起点角度
    theta_start = np.pi / 4
    theta_end = theta_start + theta_diff

    # 计算线段的起点和终点
    x_start = r_circle * np.cos(theta_start)
    y_start = r_circle * np.sin(theta_start)
    x_end = r_circle * np.cos(theta_end)
    y_end = r_circle * np.sin(theta_end)

    # 延伸线段
    extension_length = 0.275
    direction = np.array([x_end - x_start, y_end - y_start])
    direction = direction / np.linalg.norm(direction)  # 单位化方向向量
    x_extend = x_end + extension_length * direction[0]
    y_extend = y_end + extension_length * direction[1]

    # 计算垂直方向远离圆心
    perpendicular = np.array([direction[1], -direction[0]])  # 逆时针旋转90度
    x_perpendicular = x_extend + 0.15 * perpendicular[0]
    y_perpendicular = y_extend + 0.15 * perpendicular[1]

    r = np.sqrt(x_perpendicular**2 + y_perpendicular**2)
    return r


def circle_2(point):
    x, y = point[0], point[1]
    r = np.sqrt(x**2 + y**2)
    return r


if __name__ == "__main__":
    r_1 = circle_1()
    r_2 = circle_2((1.65 / 2, r_1 + 0.15))
    p = r_2 - 4.5
    print(f"近似结果的螺距为: {p}")

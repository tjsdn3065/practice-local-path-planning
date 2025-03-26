#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def load_global_path(filename):
    limit = 0
    x_points = []
    y_points = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if limit >= 200:
                break
            x_points.append(float(parts[3]) - 249333.211)
            y_points.append(float(parts[4]) - 3688202.931)
            limit += 1
    return np.array(x_points), np.array(y_points)

def compute_arc_length(x_points, y_points):
    s_vals = [0.0]
    for i in range(1, len(x_points)):
        ds = np.hypot(x_points[i] - x_points[i-1], y_points[i] - y_points[i-1])
        s_vals.append(s_vals[-1] + ds)
    return np.array(s_vals)

def map_d_to_angle(d, w, offset):
    if d >= 0:
        # d ∈ [w, 0] (왼쪽 연석에서 경로로)
        return ((w - d) / w) * math.pi
    elif d >= -offset:
        # d ∈ [0, -offset] (경로 부근, 상수값)
        return math.pi
    else:
        # d ∈ [-offset, -w-offset] (다른 경로에서 오른쪽 연석까지)
        return math.pi + ((d + offset) / w) * math.pi

def compute_U_road_modified(d, w, offset):
    x_val = map_d_to_angle(d, w, offset)
    U = 0.5 * (math.cos(x_val) + 1)
    return U

def compute_U_obs(s, d, s_i, d_i, v_e, v_i, delta_s, delta_d, epsilon=1e-3, cur_s = 10):
    if (v_e >= v_i and cur_s <= s_i) or (v_e < v_i and cur_s > s_i):
        rel_speed_term = 1.0 / abs(v_e - v_i) + epsilon
        exp_term = - rel_speed_term * ( ((s - s_i)**2) / (delta_s**2) + ((d - d_i)**2) / (delta_d**2) )
        return np.exp(exp_term)
    else:
        return 0.0

def compute_total_potential_field(s, d, w, offset, obstacles, v_e, delta_s, delta_d, epsilon=1e-3):
    U_road_val = compute_U_road_modified(d, w, offset)
    U_obs_total = 0.0
    for obs in obstacles:
        s_i, d_i, v_i = obs
        U_obs_total += compute_U_obs(s, d, s_i, d_i, v_e, v_i, delta_s, delta_d, epsilon)
    return max(U_road_val, U_obs_total)

def frenet_to_cartesian(s, d, cs_x, cs_y):
    x_c = cs_x(s)
    y_c = cs_y(s)
    dx_ds = cs_x.derivative()(s)
    dy_ds = cs_y.derivative()(s)
    norm_T = math.hypot(dx_ds, dy_ds)
    if norm_T < 1e-6:
        return x_c, y_c
    T_x = dx_ds / norm_T
    T_y = dy_ds / norm_T
    # 법선 벡터: (-T_y, T_x)
    N_x = -T_y
    N_y = T_x
    X = x_c + d * N_x
    Y = y_c + d * N_y
    return X, Y

def vectorized_frenet_to_cartesian(S, D, cs_x, cs_y):
    X = np.zeros_like(S)
    Y = np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            X[i, j], Y[i, j] = frenet_to_cartesian(S[i, j], D[i, j], cs_x, cs_y)
    return X, Y

def main():
    # (1) 전역 경로 파일 로드
    filename = "ERP_Racing_1차선_경로.txt"  # path.txt 파일은 현재 폴더에 있다고 가정
    x_points, y_points = load_global_path(filename)

    # (2) 호 길이(s) 계산 및 CubicSpline 보간 (보간 타입은 필요에 따라 수정)
    s_vals = compute_arc_length(x_points, y_points)
    total_length = s_vals[-1]
    cs_x = CubicSpline(s_vals, x_points, bc_type='natural')
    cs_y = CubicSpline(s_vals, y_points, bc_type='natural')

    # (3) 프레넷 좌표계 격자 생성: s ∈ [0, total_length], d ∈ [ -w-offset, w ]
    # 여기서 d의 범위를 사용자가 원하는 대로 설정합니다.
    # 예를 들어, w = 1.0, offset = 2.0 라고 하면,
    # d의 범위는 오른쪽 연석: -w-offset = -3.0, 경로: 0, 왼쪽 연석: w = 1.0
    w = 1.0 # 경로로 부터 연석까지 실제 측정해야함
    offset = 2.0 # 두 경로 사이의 거리 계산 값(약 2m)
    num_s = 200
    num_d = 100
    # d의 전체 범위: [w (왼쪽 연석) ~ 0 ~ -offset ~ -w-offset (오른쪽 연석)]
    d_min = -w - offset  # 예: -3.0
    d_max = w           # 예: 1.0
    s_grid = np.linspace(0, total_length, num_s)
    d_grid = np.linspace(d_max, d_min, num_d)  # d가 큰 값(왼쪽)에서 작은 값(오른쪽) 순서로
    S, D = np.meshgrid(s_grid, d_grid)

    # (4) 전위장 계산 (계산은 프레넷 좌표계에서 이루어짐)
    # 잠재력 함수 파라미터 (예시)
    v_e = 5.89     # Ego 차량 속도 (m/s, 약 50 km/h)
    delta_s = 5.0   # s 방향 위험 영역 표준편차
    delta_d = 0.5   # d 방향 위험 영역 표준편차
    epsilon = 1e-3

    # 장애물 설정: 예시로, 전역 경로 상의 특정 s 위치에 장애물이 있다고 가정.
    obstacles = [
        (15, 0.0, 0.0),   # 정적 장애물
        (5, -2.0, 8.33)     # 움직이는 장애물 (약 30 km/h)
    ]

    U_total = np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            s_val = S[i, j]
            d_val = D[i, j]
            U_total[i, j] = compute_total_potential_field(s_val, d_val, w, offset,
                                                          obstacles, v_e, delta_s, delta_d, epsilon)

    # (5) 프레넷 격자 (S, D)를 카테시안 좌표계로 변환 (보간 함수 cs_x, cs_y 사용)
    X, Y = vectorized_frenet_to_cartesian(S, D, cs_x, cs_y)

    # (6) 최종 출력: 카테시안 좌표계에서 전위장을 contour plot과 3D surface plot으로 시각화
    fig = plt.figure(figsize=(12, 6))

    # Contour Plot (Cartesian)
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(X, Y, U_total, levels=50, cmap='viridis')
    ax1.set_title("Modified Potential Field Contour (Cartesian Coordinates)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    fig.colorbar(contour, ax=ax1)

    # 3D Surface Plot (Cartesian)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, U_total, cmap='viridis', edgecolor='none')
    ax2.set_title("Modified Potential Field Surface (Cartesian Coordinates)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("U_total")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

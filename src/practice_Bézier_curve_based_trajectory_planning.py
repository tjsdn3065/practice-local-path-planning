#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import simps

# =============================================================================
# 1. 좌표 변환 함수
# =============================================================================
def to_global(x, y, x0, y0, theta0):
    T = np.array([
        [np.cos(theta0), -np.sin(theta0), x0],
        [np.sin(theta0),  np.cos(theta0), y0],
        [0,              0,              1]
    ])
    local_point = np.array([x, y, 1])
    global_point = T.dot(local_point)
    return global_point[0], global_point[1]

def to_local(x, y, x0, y0, theta0):
    T = np.array([
        [np.cos(theta0), -np.sin(theta0), x0],
        [np.sin(theta0),  np.cos(theta0), y0],
        [0,              0,              1]
    ])
    T_inv = np.linalg.inv(T)
    global_point = np.array([x, y, 1])
    local_point = T_inv.dot(global_point)
    return local_point[0], local_point[1]

# =============================================================================
# 2. 전역 경로 로드 및 보간
# =============================================================================
def load_global_path(filename):
    limit = 0
    x_points, y_points = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if limit >= 300:
                break
            # 파일의 4번째, 5번째 컬럼을 사용, 기준점 보정
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

def global_curvature(cs_x, cs_y, s):
    dx_ds = cs_x.derivative()(s)
    dy_ds = cs_y.derivative()(s)
    d2x_ds2 = cs_x.derivative(2)(s)
    d2y_ds2 = cs_y.derivative(2)(s)
    numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
    denominator = (dx_ds**2 + dy_ds**2)**1.5
    return 0.0 if abs(denominator) < 1e-6 else numerator / denominator

# =============================================================================
# 3. 전역 경로 상에서의 s 찾기 및 횡 오프셋 계산
# =============================================================================
def find_closest_s_newton(x0, y0, cs_x, cs_y, s_candidates, total_length):
    xs = cs_x(s_candidates)
    ys = cs_y(s_candidates)
    distances = (x0 - xs)**2 + (y0 - ys)**2
    s_current = s_candidates[np.argmin(distances)]
    max_iter, tol = 30, 1e-6
    for _ in range(max_iter):
        fprime = dist_sq_grad(s_current, x0, y0, cs_x, cs_y)
        fsecond = dist_sq_hess(s_current, x0, y0, cs_x, cs_y)
        if abs(fsecond) < 1e-12:
            break
        step = -fprime / fsecond
        s_current = (s_current + step) % total_length
        if abs(step) < tol:
            break
    return s_current

def dist_sq_grad(s, x0, y0, cs_x, cs_y):
    dx = x0 - cs_x(s)
    dy = y0 - cs_y(s)
    dxds = cs_x.derivative()(s)
    dyds = cs_y.derivative()(s)
    return -2.0 * (dx * dxds + dy * dyds)

def dist_sq_hess(s, x0, y0, cs_x, cs_y):
    dx = x0 - cs_x(s)
    dy = y0 - cs_y(s)
    dxds = cs_x.derivative()(s)
    dyds = cs_y.derivative()(s)
    d2xds2 = cs_x.derivative(2)(s)
    d2yds2 = cs_y.derivative(2)(s)
    val = (-dxds**2 + dx * d2xds2) + (-dyds**2 + dy * d2yds2)
    return -2.0 * val

def signed_lateral_offset(x0, y0, s0, cs_x, cs_y):
    px, py = cs_x(s0), cs_y(s0)
    tx, ty = cs_x.derivative()(s0), cs_y.derivative()(s0)
    dx_veh, dy_veh = x0 - px, y0 - py
    cross_val = tx * dy_veh - ty * dx_veh
    q0 = math.sqrt(dx_veh**2 + dy_veh**2)
    return q0 if cross_val > 0.0 else -q0

# =============================================================================
# 4. 위험 잠재력 필드 (프레넷 좌표계 기반)
# =============================================================================
def map_d_to_angle(d, w, offset):
    if d >= 0:
        return ((w - d) / w) * math.pi
    elif d >= -offset:
        return math.pi
    else:
        return math.pi + ((d + offset) / w) * math.pi

def compute_U_road_modified(d, w, offset):
    x_val = map_d_to_angle(d, w, offset)
    return 0.5 * (math.cos(x_val) + 1)

def compute_U_obs(s, d, s_i, d_i, v_e, v_i, delta_s, delta_d, epsilon=1e-3, cur_s=25):
    if (v_e >= v_i and cur_s <= s_i) or (v_e < v_i and cur_s > s_i):
        rel_speed_term = 1.0 / abs(v_e - v_i) + epsilon
        exp_term = - rel_speed_term * (((s - s_i)**2)/(delta_s**2) + ((d - d_i)**2)/(delta_d**2))
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

# =============================================================================
# 5. Frenet -> Cartesian 변환
# =============================================================================
def frenet_to_cartesian(s, d, cs_x, cs_y):
    x_c, y_c = cs_x(s), cs_y(s)
    dx_ds, dy_ds = cs_x.derivative()(s), cs_y.derivative()(s)
    norm_T = math.hypot(dx_ds, dy_ds)
    if norm_T < 1e-6:
        return x_c, y_c
    T_x, T_y = dx_ds / norm_T, dy_ds / norm_T
    N_x, N_y = -T_y, T_x
    return x_c + d * N_x, y_c + d * N_y

def vectorized_frenet_to_cartesian(S, D, cs_x, cs_y):
    X = np.zeros_like(S)
    Y = np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            X[i, j], Y[i, j] = frenet_to_cartesian(S[i, j], D[i, j], cs_x, cs_y)
    return X, Y

# =============================================================================
# 6. 4차 Bézier 곡선 관련 함수
# =============================================================================
def bernstein_poly(i, n, tau):
    return math.comb(n, i) * (tau**i) * ((1 - tau)**(n - i))

def bezier_curve(control_points, tau):
    n = 4
    point = np.zeros(2)
    for i in range(n + 1):
        point += bernstein_poly(i, n, tau) * np.array(control_points[i])
    return point

def bezier_derivative(control_points, tau):
    n = 4
    d_point = np.zeros(2)
    for i in range(n):
        d_point += math.comb(n - 1, i) * (tau**i) * ((1 - tau)**(n - 1 - i)) * (control_points[i + 1] - control_points[i]) * n
    return d_point

def bezier_second_derivative(control_points, tau):
    n = 4
    dd_point = np.zeros(2)
    for i in range(n - 1):
        dd_point += math.comb(n - 2, i) * (tau**i) * ((1 - tau)**(n - 2 - i)) * (control_points[i + 2] - 2 * control_points[i + 1] + control_points[i]) * (n * (n - 1))
    return dd_point

def curvature(control_points, tau):
    d1 = bezier_derivative(control_points, tau)
    d2 = bezier_second_derivative(control_points, tau)
    num = d1[0] * d2[1] - d1[1] * d2[0]
    denom = (d1[0]**2 + d1[1]**2)**1.5
    return 0.0 if abs(denom) < 1e-6 else num / denom

def bezier_arc_length(control_points, num_points=100):
    taus = np.linspace(0, 1, num_points)
    pts = np.array([bezier_curve(control_points, tau) for tau in taus])
    length = 0.0
    for i in range(1, len(pts)):
        length += np.linalg.norm(pts[i] - pts[i - 1])
    return length

# =============================================================================
# 7. 제어점 생성 (식 23~26)
# =============================================================================
def generate_control_points(params, x0, y0, x_t, y_t, s_current, s_target, theta0, theta_t, kappa):
    """
    새로운 좌표계에서 제어점을 생성한 후 전역 좌표계로 변환합니다.

    입력:
      params: [s0, x2, s4] (독립 변수)
      x0, y0: 차량의 현재 전역 위치
      x_t, y_t: 목표 전역 위치
      s_current, s_target: 현재 및 목표 s 값 (호 길이)
      theta0, theta_t: 현재 및 목표 헤딩 (라디안)
      kappa: 초기 곡률
    출력:
      5개의 제어점 [P0, P1, P2, P3, P4] (각각 전역 좌표계 상의 [x, y])
    """
    s0, x2, s4 = params
    # 로컬 좌표계에서 P0는 (0,0)
    P0 = np.array([0.0, 0.0])
    P1 = np.array([s0, 0.0])
    # P2: y 성분은 (4 * kappa * s0^2) / 3, x 성분은 자유 변수 x2
    P2 = np.array([x2, (4 * kappa * (s0**2)) / 3.0])

    # 전역 목표 상태를 로컬 좌표계로 변환 (회전행렬 이용)
    # T = np.array([[np.cos(theta0), -np.sin(theta0)],
    #               [np.sin(theta0),  np.cos(theta0)]])
    # p4_input = np.array([x_t - x0, y_t - y0])
    # P4_local = T.dot(p4_input)
    # P4 = np.array([P4_local[0], P4_local[1]])

    # # P3: 목표 상태 P4에서 s4만큼 뒤로 이동 (목표 헤딩 theta_t를 반영하여)
    # P3 = np.array([P4[0] - s4 * np.cos(theta_t - theta0),
    #                P4[1] - s4 * np.sin(theta_t - theta0)])

    global_P4 = np.array([x_t, y_t])
    local_P4 = np.array(to_local(global_P4[0], global_P4[1]))
    P3 = np.array([local_P4[0] - s4 * np.cos(theta_t - theta0), local_P4[1] - s4 * np.sin(theta_t - theta0)])

    # 로컬 좌표계에서 계산된 제어점들을 전역 좌표계로 변환
    global_P0 = np.array(to_global(P0[0], P0[1], x0, y0, theta0))
    global_P1 = np.array(to_global(P1[0], P1[1], x0, y0, theta0))
    global_P2 = np.array(to_global(P2[0], P2[1], x0, y0, theta0))
    global_P3 = np.array(to_global(P3[0], P3[1], x0, y0, theta0))
    #global_P4 = np.array(to_global(P4[0], P4[1], x0, y0, theta0))

    return [global_P0, global_P1, global_P2, global_P3, global_P4]

# =============================================================================
# 8. 비용 함수 (식 27)
# =============================================================================
def U_total_from_field(control_points, tau, w, offset, obstacles, v_e, delta_s, delta_d, epsilon, cs_x, cs_y, s_candidates, total_length):
    point = bezier_curve(control_points, tau)
    x, y = point[0], point[1]
    s_val = find_closest_s_newton(x, y, cs_x, cs_y, s_candidates, total_length)
    d_val = signed_lateral_offset(x, y, s_val, cs_x, cs_y)
    return compute_total_potential_field(s_val, d_val, w, offset, obstacles, v_e, delta_s, delta_d, epsilon)

def cost_function_field_new(params, x0, y0, x_t, y_t, s_current, s_target, theta0, theta_t, kappa,
                            w, offset, obstacles, v_e, delta_s, delta_d, epsilon, num_points, cs_x, cs_y, s_candidates, total_length):
    control_points = generate_control_points(params, x0, y0, x_t, y_t, s_current, s_target, theta0, theta_t, kappa)
    taus = np.linspace(0, 1, num_points)
    curvatures = np.array([curvature(control_points, tau) for tau in taus])
    U_vals = np.array([U_total_from_field(control_points, tau, w, offset, obstacles, v_e, delta_s, delta_d, epsilon, cs_x, cs_y, s_candidates, total_length) for tau in taus])
    J_curv = simps(curvatures**2, taus)
    J_U = simps(U_vals, taus)
    return J_curv + J_U

# =============================================================================
# 9. 속도 프로파일 최적화 (식 31~35 보충)
# =============================================================================
def velocity_profile(v0, a0, T, s_T):
    """2차 다항식 속도 함수: v(t) = A t^2 + a0 t + v0, A는 s_T 조건에 따라 결정"""
    A_coef = (6 * s_T - 3 * a0 * T**2 - 6 * v0 * T) / (2 * T**3)
    ts = np.linspace(0, T, 100)
    v_t = A_coef * ts**2 + a0 * ts + v0
    a_t = 2 * A_coef * ts + a0
    return ts, v_t, a_t

def optimize_velocity_profile(s_T, v0, a0, v_max, a_min, a_max, T_bounds=(0.1, 10.0)):
    """
    주어진 경로 길이 s_T, 초기 속도 v0, 초기 가속도 a0에 대해,
    2차 속도 함수 v(t)=A t^2 + a0 t + v0 (A는 s_T 조건에 따라 결정됨)
    가속도 및 속도 제약:
       0 ≤ v(t) ≤ v_max,  a_min ≤ a(t) ≤ a_max,  for t in [0, T]
    조건을 만족하면서 주행시간 T를 최소화 (즉, 평균 속도 s_T/T 최대화)
    최적 T를 찾아 반환합니다.
    """
    def objective(T):
        # T가 너무 작으면 비현실적이므로 큰 페널티 부여
        if T < T_bounds[0] or T > T_bounds[1]:
            return 1e6
        A_coef = (6 * s_T - 3 * a0 * T**2 - 6 * v0 * T) / (2 * T**3)
        ts = np.linspace(0, T, 100)
        v_t = A_coef * ts**2 + a0 * ts + v0
        a_t = 2 * A_coef * ts + a0
        # 제약 조건 위반 시 페널티
        penalty = 0.0
        if np.any(v_t < 0) or np.any(v_t > v_max):
            penalty += 1e4
        if np.any(a_t < a_min) or np.any(a_t > a_max):
            penalty += 1e4
        # 평균 속도 = s_T / T (최소 T가 목표이므로 objective는 T)
        return T + penalty
    res = minimize_scalar(objective, bounds=T_bounds, method='bounded')
    T_opt = res.x
    A_coef = (6 * s_T - 3 * a0 * T_opt**2 - 6 * v0 * T_opt) / (2 * T_opt**3)
    ts = np.linspace(0, T_opt, 100)
    v_t = A_coef * ts**2 + a0 * ts + v0
    a_t = 2 * A_coef * ts + a0
    return T_opt, ts, v_t, a_t

# =============================================================================
# 10. 후보 Bézier 곡선 클러스터 생성 (옵션)
# =============================================================================
def generate_candidate_cluster(opt_params, x0, y0, x_t, y_t, s_current, s_target, theta0, theta_t, kappa, num_candidates=10):
    candidate_curves = []
    candidate_control_points = []
    np.random.seed(0)
    taus = np.linspace(0, 1, 100)
    for i in range(num_candidates):
        perturbation = np.random.uniform(-0.5, 0.5, size=3)
        candidate_params = opt_params + perturbation
        cp = generate_control_points(candidate_params, x0, y0, x_t, y_t, s_current, s_target, theta0, theta_t, kappa)
        candidate_control_points.append(cp)
        curve = np.array([bezier_curve(cp, tau) for tau in taus])
        candidate_curves.append(curve)
    return candidate_curves, candidate_control_points

# =============================================================================
# 11. 목표 지점에서 lateral 포텐셜 프로파일 시각화
# =============================================================================
def plot_target_potential_profile(s_target, w, offset, obstacles, v_e, delta_s, delta_d, epsilon=1e-3):
    d_values = np.linspace(-w - offset, w, 200)
    U_vals = [compute_total_potential_field(s_target, d, w, offset, obstacles, v_e, delta_s, delta_d, epsilon) for d in d_values]
    plt.figure(figsize=(8,4))
    plt.plot(d_values, U_vals, 'r-', linewidth=2)
    plt.xlabel("Lateral coordinate d (m)")
    plt.ylabel("Potential U_total")
    plt.title(f"Potential Field Profile at Target s = {s_target:.2f} m")
    plt.grid(True)
    plt.show()
    return d_values, U_vals

# =============================================================================
# 12. 메인 함수: 전체 통합 실행 및 시각화
# =============================================================================
def main():
    # 전역 경로 로드 및 보간
    filename = "ERP_Racing_1차선_경로.txt"
    x_points, y_points = load_global_path(filename)
    s_vals = compute_arc_length(x_points, y_points)
    total_length = s_vals[-1]
    num_samples = 3000
    s_candidates = np.linspace(0, total_length, num_samples)
    cs_x = CubicSpline(s_vals, x_points, bc_type='natural')
    cs_y = CubicSpline(s_vals, y_points, bc_type='natural')

    # 현재 차량 상태 (예: s_current = 25)
    s_current = 25.0
    x0 = cs_x(s_current)
    y0 = cs_y(s_current)
    theta0 = math.atan2(cs_y.derivative()(s_current), cs_x.derivative()(s_current))
    kappa0 = global_curvature(cs_x, cs_y, s_current)
    print("Global current state (x0, y0, θ0):", (x0, y0, theta0))
    print("Computed initial curvature at s=25:", kappa0)

    # 목표 상태 (예: s_target = 35)
    s_target = 35.0
    x_t = cs_x(s_target)
    y_t = cs_y(s_target)
    theta_t = math.atan2(cs_y.derivative()(s_target), cs_x.derivative()(s_target))

    # 최적화: 최적 제어점을 찾기 위한 비용 함수 최소화 (독립 변수: [s0, x2, s4])
    init_params = [1.0, 0.0, 1.0]
    w = 1.0       # 연석까지의 거리
    offset = 2.0  # 두 경로 사이 간격
    obstacles = [] #[(30, 0.0, 0.0), (20, -2.0, 8.33)]
    v_e = 15.89    # Ego 차량 속도 (m/s)
    delta_s = 1.0
    delta_d = 0.5
    epsilon = 1e-3
    num_points = 100

    res = minimize(cost_function_field_new, init_params,
                   args=(x0, y0, x_t, y_t, s_current, s_target, theta0, theta_t, kappa0,
                         w, offset, obstacles, v_e, delta_s, delta_d, epsilon, num_points, cs_x, cs_y, s_candidates, total_length),
                   bounds=[(0.1, 10), (0.1, 10), (0.1, 10)],
                   method='SLSQP')
    print("Optimized parameters [s0, x2, s4]:", res.x)
    optimal_control_points = generate_control_points(res.x, x0, y0, x_t, y_t, s_current, s_target, theta0, theta_t, kappa0)
    print("Optimal Control Points:", optimal_control_points)
    taus = np.linspace(0, 1, 100)
    optimal_curve = np.array([bezier_curve(optimal_control_points, tau) for tau in taus])

    # 후보 클러스터 생성 (옵션)
    candidate_curves, candidate_control_points = generate_candidate_cluster(res.x, x0, y0, x_t, y_t, s_current, s_target, theta0, theta_t, kappa0, num_candidates=10)

    # 전역 경로 일부 시각화 (예: s=0~50)
    global_s_segment = np.linspace(0, 50, 100)
    global_x = cs_x(global_s_segment)
    global_y = cs_y(global_s_segment)

    # 최적 Bézier 곡선의 호 길이 계산
    s_T = bezier_arc_length(optimal_control_points)
    print("Arc length of optimal Bézier curve (s_T):", s_T)

    # 속도 프로파일 최적화 (식 35 최적화 문제 구현)
    # 여기서는 속도 제한과 가속도 제한을 상수로 두고 최적 T를 찾습니다.
    v0 = 15.0    # 초기 속도 (m/s)
    a0 = 0.0     # 초기 가속도 (m/s²)
    v_max = 20.0 # 최대 속도 제한 (m/s)
    a_min = -3.0 # 최소 가속도
    a_max = 3.0  # 최대 가속도
    T_opt, ts, v_profile, a_profile = optimize_velocity_profile(s_T, v0, a0, v_max, a_min, a_max, T_bounds=(0.1, 10.0))
    print("Optimized travel time T:", T_opt)

    # ----------------------------------------------------------------
    # 시각화
    # ----------------------------------------------------------------
    plt.figure(figsize=(10,6))
    # 전역 경로, 최적 Bézier 곡선, 후보 클러스터, 제어점 표시
    for curve in candidate_curves:
        plt.plot(curve[:,0], curve[:,1], color='gray', linewidth=1, alpha=0.6)
    plt.plot(optimal_curve[:,0], optimal_curve[:,1], color='red', linewidth=3, label='Optimal Bézier Path')
    opt_cp = np.array(optimal_control_points)
    labels = ["P0", "P1", "P2", "P3", "P4"]
    for i, cp in enumerate(opt_cp):
        plt.plot(cp[0], cp[1], 'bo', markersize=8)
        plt.text(cp[0], cp[1], labels[i], fontsize=12, color='blue',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    plt.plot(global_x, global_y, 'k--', linewidth=2, label='Global Path')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Optimal Bézier Path, Candidate Cluster & Global Path")
    plt.legend()
    plt.grid(True)

    # 곡률 및 위험 잠재력 플롯 (로컬 매개변수 τ)
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    local_curvatures = np.array([curvature(optimal_control_points, tau) for tau in taus])
    plt.plot(taus, local_curvatures, 'b-')
    plt.title("Curvature along the Optimal Bézier Path")
    plt.ylabel("Curvature (1/m)")
    plt.grid(True)
    plt.subplot(2,1,2)
    U_vals_field = np.array([U_total_from_field(optimal_control_points, tau, w, offset, obstacles, v_e, delta_s, delta_d, epsilon, cs_x, cs_y, s_candidates, total_length) for tau in taus])
    plt.plot(taus, U_vals_field, 'g-')
    plt.title("Potential Field U_total along the Optimal Bézier Path")
    plt.xlabel("τ")
    plt.ylabel("U_total")
    plt.grid(True)

    # 속도 및 가속도 프로파일 플롯
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(ts, v_profile, 'm-', label='Velocity')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity Profile")
    plt.legend()
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(ts, a_profile, 'c-', label='Acceleration')
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Acceleration Profile")
    plt.legend()
    plt.grid(True)

    # 목표 지점에서의 lateral 포텐셜 프로파일 시각화
    plot_target_potential_profile(s_target, w, offset, obstacles, v_e, delta_s, delta_d, epsilon)

    # 전위장 필드 계산 및 3D 시각화 (전체 s 범위)
    num_s = 3000
    s_range = np.linspace(0, total_length, num_s)
    d_range = np.linspace(-w - offset, w, 200)
    S, D = np.meshgrid(s_range, d_range)
    U_total_field = np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            s_val = S[i, j]
            d_val = D[i, j]
            U_total_field[i, j] = compute_total_potential_field(s_val, d_val, w, offset, obstacles, v_e, delta_s, delta_d, epsilon)
    X_field, Y_field = vectorized_frenet_to_cartesian(S, D, cs_x, cs_y)
    fig_field = plt.figure(figsize=(12,6))
    ax_field1 = fig_field.add_subplot(1,2,1)
    contour = ax_field1.contourf(X_field, Y_field, U_total_field, levels=50, cmap='viridis')
    ax_field1.set_title("Modified Potential Field Contour")
    ax_field1.set_xlabel("X (m)")
    ax_field1.set_ylabel("Y (m)")
    fig_field.colorbar(contour, ax=ax_field1)
    ax_field2 = fig_field.add_subplot(1,2,2, projection='3d')
    ax_field2.plot_surface(X_field, Y_field, U_total_field, cmap='viridis', edgecolor='none')
    ax_field2.set_title("Modified Potential Field Surface")
    ax_field2.set_xlabel("X (m)")
    ax_field2.set_ylabel("Y (m)")
    ax_field2.set_zlabel("U_total")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

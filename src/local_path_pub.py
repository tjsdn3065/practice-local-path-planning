#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import math
import numpy as np
from math import pi, hypot
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion
from morai_msgs.msg import EgoVehicleStatus, CandidatePaths, ObjectStatusList

class GlobalPathAnalyzer:
    def __init__(self):
        rospy.init_node('global_path_analyzer', anonymous=True)

        self.s_candidates = None

        # 전역 경로 보간 관련 변수
        self.cs_x = None
        self.cs_y = None
        self.s_vals = None
        self.total_length = None
        self.is_global_path_ready = False

        # 오도메트리 수신 여부 및 위치/헤딩
        self.is_odom_received = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0   # 라디안
        self.current_speed = 0.0

        # 장애물 정보
        self.is_obstacle  =False
        self.obstacles_s = []

        # 이전 루프에서 계산한 s 값
        self.prev_s = None

        # 디버그 및 로컬 경로 발행
        # self.debug_pub = rospy.Publisher('/debug_s0_q0', Float32MultiArray, queue_size=1)
        # 커스텀 메시지 CandidatePaths 퍼블리셔
        self.candidate_path_pub = rospy.Publisher('/candidate_paths', CandidatePaths, queue_size=1)
        self.candidate_path_pub1 = rospy.Publisher('/candidate_path1', Path, queue_size=1)
        self.candidate_path_pub2 = rospy.Publisher('/candidate_path2', Path, queue_size=1)
        self.candidate_path_pub3 = rospy.Publisher('/candidate_path3', Path, queue_size=1)
        self.candidate_path_pub4 = rospy.Publisher('/candidate_path4', Path, queue_size=1)
        self.candidate_path_pub5 = rospy.Publisher('/candidate_path5', Path, queue_size=1)
        self.candidate_path_pub6 = rospy.Publisher('/candidate_path6', Path, queue_size=1)
        self.candidate_path_pub7 = rospy.Publisher('/candidate_path7', Path, queue_size=1)

        # 토픽 구독
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        rospy.Subscriber("/Object_topic", ObjectStatusList, self.obstacle_callback)
        rospy.Subscriber("Ego_topic", EgoVehicleStatus, self.status_callback)

        rate = rospy.Rate(20)  # 20Hz
        while not rospy.is_shutdown():
            os.system('clear')
            if self.is_odom_received and self.is_global_path_ready and self.is_status and self.is_obstacle:
                """
                주어진 전역 좌표 (x, y)에 대해, 전역 경로 상의 s 값을 벡터화된 방법으로 계산합니다.
                """
                # 만약 self.s_candidates가 존재하지 않으면 미리 계산해둡니다.
                if not hasattr(self, 's_candidates') or self.s_candidates is None:
                    num_samples = 1000  # 필요에 따라 조절
                    self.s_candidates = np.linspace(0, self.total_length, num_samples)
                # 1) 현재 위치로부터 s0 계산
                s0 = self.find_closest_s_newton(self.current_x, self.current_y)
                # 2) 현재 횡방향 오프셋 q0 계산
                q0 = self.signed_lateral_offset(self.current_x, self.current_y, s0)

                # 3) 로컬 경로 생성 (Δs는 차량 속도 및 장애물 정보 반영)
                # 후보 경로들을 생성 (-1.5부터 1.5까지 0.1 단위)
                candidate_paths_list = self.generate_candidate_paths(s0, q0)

                # CandidatePaths 커스텀 메시지에 모든 후보 경로 저장
                candidate_paths_msg = CandidatePaths()
                candidate_paths_msg.paths = candidate_paths_list

                # 최적 경로 선택
                optimal_path = self.compute_optimal_path(candidate_paths_msg.paths)

                # 퍼블리시 (한 번에 모든 후보 경로 전달)
                self.candidate_path_pub.publish(candidate_paths_msg)
                self.candidate_path_pub1.publish(candidate_paths_msg.paths[0])
                self.candidate_path_pub2.publish(candidate_paths_msg.paths[1])
                self.candidate_path_pub3.publish(candidate_paths_msg.paths[2])
                self.candidate_path_pub4.publish(candidate_paths_msg.paths[3])
                self.candidate_path_pub5.publish(candidate_paths_msg.paths[4])
                self.candidate_path_pub6.publish(candidate_paths_msg.paths[5])
                self.candidate_path_pub7.publish(candidate_paths_msg.paths[6])
                candidate_path = candidate_paths_msg.paths[6]
                # print("Candidate Path 6:")
                # for i, pose in enumerate(candidate_path.poses):
                #     x = pose.pose.position.x
                #     y = pose.pose.position.y
                #     print("Point {}: x = {:.3f}, y = {:.3f}".format(i, x, y))

                # 디버그 출력
                x_s0 = self.cs_x(s0)
                y_s0 = self.cs_y(s0)
                rospy.loginfo("차량 (x,y)=(%.3f, %.3f), yaw=%.3f, speed=%.2f -> s0=%.3f, 경로 좌표=(%.3f, %.3f), q0=%.3f",
                              self.current_x, self.current_y, self.current_yaw * 180/pi, self.current_speed,
                              s0, x_s0, y_s0, q0)
                rospy.loginfo("장애물 좌표 업데이트: %s", self.obstacles_s)

                # debug_msg = Float32MultiArray()
                # debug_msg.data = [s0, q0]
                # self.debug_pub.publish(debug_msg)

                self.is_odom_received = False
                # self.is_global_path_ready = False
                self.is_status = False
                self.is_obstacle = False

            rate.sleep()


    def status_callback(self, msg):
        self.is_status = True
        self.current_speed = msg.velocity.x * 3.75

    # ---------------------------------------------------------
    # 전역 경로 /global_path 콜백
    # ---------------------------------------------------------
    def global_path_callback(self, msg):
        if not msg.poses:
            rospy.logwarn("수신한 global_path가 비어 있습니다.")
            return

        if self.is_global_path_ready is False:
            x_points = []
            y_points = []
            for pose_stamped in msg.poses:
                x_points.append(pose_stamped.pose.position.x)
                y_points.append(pose_stamped.pose.position.y)

            # 폐곡선 가정: 첫번째와 마지막 좌표가 같도록 보정
            tol = 1e-8
            if abs(x_points[0] - x_points[-1]) > tol or abs(y_points[0] - y_points[-1]) > tol:
                rospy.logwarn("첫번째와 마지막 좌표가 다릅니다. 강제로 보정합니다.")
                x_points[-1] = x_points[0]
                y_points[-1] = y_points[0]

            # 호 길이 s_vals 계산
            s_vals = [0.0]
            for i in range(1, len(x_points)):
                dist = hypot(x_points[i] - x_points[i-1],
                            y_points[i] - y_points[i-1])
                s_vals.append(s_vals[-1] + dist)

            self.s_vals = s_vals
            self.total_length = s_vals[-1]

            # 주기적 스플라인 보간
            self.cs_x = CubicSpline(self.s_vals, x_points, bc_type='periodic')
            self.cs_y = CubicSpline(self.s_vals, y_points, bc_type='periodic')

            self.is_global_path_ready = True
            # rospy.loginfo("글로벌 경로 보간 완료. 전체 길이=%.3f", self.total_length)

    # ---------------------------------------------------------
    # 오도메트리 /odom 콜백 (위치, 헤딩, 속도 업데이트)
    # ---------------------------------------------------------
    def odom_callback(self, msg):
        self.is_odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # 헤딩 (yaw) 계산
        orientation_q = msg.pose.pose.orientation
        quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        _, _, yaw = euler_from_quaternion(quaternion)
        self.current_yaw = yaw

    # ---------------------------------------------------------
    # 장애물 정보
    # ---------------------------------------------------------
    def obstacle_callback(self, msg):
        """
        ObjectStatusList 메시지에서 obstacle_list에 있는 각 장애물의
        위치를 이용하여, 전역 경로 상의 s 좌표를 계산하고 self.obstacles_s에 업데이트합니다.
        """
        if not self.is_global_path_ready:
            rospy.logwarn("전역 경로가 아직 준비되지 않았습니다.")
            return
        self.is_obstacle = True
        obstacles_s = []

        for obstacle in msg.obstacle_list:
            x_obs = obstacle.position.x
            y_obs = obstacle.position.y
            s_val, q_val = self.compute_obstacle_frenet(x_obs, y_obs)
            # q_obs가 1.5보다 큰 경우 제외
            if abs(q_val) <= 1.5:
                obstacles_s.append((s_val, q_val))
        self.obstacles_s = obstacles_s

    def compute_obstacle_frenet(self, x_obs, y_obs):
        s_obs = self.compute_s_coordinate(x_obs, y_obs)
        q_obs = self.signed_lateral_offset(x_obs, y_obs, s_obs)
        return s_obs, q_obs

    def compute_s_coordinate(self, x, y):
        # 전역 경로의 x, y 좌표를 한 번에 계산 (벡터화)
        xs = self.cs_x(self.s_candidates)
        ys = self.cs_y(self.s_candidates)

        # 장애물 위치와의 거리 제곱을 벡터화하여 계산
        distances = (x - xs)**2 + (y - ys)**2
        s_best = self.s_candidates[np.argmin(distances)]
        return s_best

    # ---------------------------------------------------------
    # 1) s0 찾기 (2차 최소화와 뉴튼 방법)
    # ---------------------------------------------------------
    def find_closest_s_newton(self, x0, y0):
        if not self.is_global_path_ready:
            return 0.0

        if self.prev_s is not None:
            s_current = self.prev_s
        else:
            # CubicSpline은 벡터화되어 있어, 한 번에 여러 s 값을 평가할 수 있음
            xs = self.cs_x(self.s_candidates)
            ys = self.cs_y(self.s_candidates)
            distances = (x0 - xs)**2 + (y0 - ys)**2  # 벡터화된 거리 계산
            s_current = self.s_candidates[np.argmin(distances)]

        max_iter = 30
        tol = 1e-6
        for _ in range(max_iter):
            fprime = self.dist_sq_grad(s_current, x0, y0)
            fsecond = self.dist_sq_hess(s_current, x0, y0)
            if abs(fsecond) < 1e-12:
                break
            step = -fprime / fsecond
            s_current += step
            s_current = s_current % self.total_length
            if abs(step) < tol:
                break
        self.prev_s = s_current
        return s_current

    def dist_sq_grad(self, s, x0, y0):
        dx = x0 - self.cs_x(s)
        dy = y0 - self.cs_y(s)
        dxds = self.cs_x.derivative()(s)
        dyds = self.cs_y.derivative()(s)
        return -2.0 * (dx * dxds + dy * dyds)

    def dist_sq_hess(self, s, x0, y0):
        dx = x0 - self.cs_x(s)
        dy = y0 - self.cs_y(s)
        dxds = self.cs_x.derivative()(s)
        dyds = self.cs_y.derivative()(s)
        d2xds2 = self.cs_x.derivative(2)(s)
        d2yds2 = self.cs_y.derivative(2)(s)
        val = (-dxds**2 + dx * d2xds2) + (-dyds**2 + dy * d2yds2)
        return -2.0 * val

    # ---------------------------------------------------------
    # 2) 현재 횡방향 오프셋 q0
    # ---------------------------------------------------------
    def signed_lateral_offset(self, x0, y0, s0):
        px = self.cs_x(s0)
        py = self.cs_y(s0)
        tx = self.cs_x.derivative()(s0)
        ty = self.cs_y.derivative()(s0)
        dx_veh = x0 - px
        dy_veh = y0 - py
        cross_val = tx * dy_veh - ty * dx_veh
        q0 = math.sqrt(dx_veh**2 + dy_veh**2)
        return +q0 if cross_val > 0.0 else -q0

    # ---------------------------------------------------------
    # (C) Δs 결정: 속도 및 장애물 정보를 반영 (식 (6) 및 Algorithm 2)
    # ---------------------------------------------------------
    def compute_delta_s_vel(self, v, a_min=-3.0, s_min=10.0, s_max=20.0):
        """
        논문 식 (6)에 따라 차량 속도 v (m/s), 최소 가속도 a_min,
        최소/최대 호 길이(s_min, s_max)를 고려해 Δs를 계산.
        """
        s_candidate = s_min + (v**2 / abs(a_min))
        if s_candidate < s_max:
            return s_candidate
        else:
            return s_max

    def compute_delta_s_with_obstacles(self, s_vehicle, s_vel, s_min=10.0):
        """
        장애물이 있을 경우, 차량 앞쪽 장애물까지의 s 거리 중 최소값과 s_vel, s_min 중 최솟값을 사용.
        장애물이 없다면 s_vel 반환.
        """
        dist_candidates = []
        for obs in self.obstacles_s:  # obs는 (s, q) 튜플
            s_obs = obs[0]
            # 장애물이 차량 앞에 있으면
            dist = s_obs - s_vehicle
            if 0 < dist < 10:
                dist_candidates.append(dist)
        if len(dist_candidates) == 0:
            return s_vel
        else:
            s_obs = min(dist_candidates)
            return min(s_obs, s_min)

    def normalize_angle(self, angle):
        """
        각도를 -pi에서 pi 사이로 정규화합니다.
        입력 각도는 radian 단위여야 합니다.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # ---------------------------------------------------------
    # 3) 로컬 경로(후보 경로) 생성 (3차 스플라인 q(s) 이용)
    # ---------------------------------------------------------
    def generate_candidate_paths(self, s0, q0):
        """
        s0: 시작 호 길이 (현재 차량 위치에 해당)
        q0: 현재 횡방향 오프셋 (signed_lateral_offset 함수로 구한 값)
        이 함수를 통해 -1.5부터 1.5까지 0.1 간격의 후보 경로들을 생성합니다.
        """
        candidate_paths = []
        for lane_offset in np.arange(-1.5, 1.5 + 0.001, 0.5):
            path_msg = self.generate_local_path(s0, q0, lane_offset)
            candidate_paths.append(path_msg)
        return candidate_paths

    def generate_local_path(self, s0, q0, lane_offset):
        """
        s0: 시작 호 길이 (현재 차량 위치에 해당하는 s)
        q0: 현재 횡방향 오프셋 (signed_lateral_offset 함수로 구한 값)
        lane_offset: 후보로 사용할 최종 오프셋 (q_f) 값 (예: -1.5, -1.4, …, 1.5)
        """
        path_msg = Path()
        path_msg.header.frame_id = "map"

        # (a) Δs 결정: 차량 속도와 장애물 정보를 반영하여 Δs를 구함
        s_vel = self.compute_delta_s_vel(self.current_speed, a_min = -3.0, s_min=10.0)
        final_delta_s = self.compute_delta_s_with_obstacles(s0, s_vel, s_min=10.0)

        # (b) 경계 조건:
        # 시작점: q(s0)=q0, q'(s0)=tan(Δθ)
        path_yaw = math.atan2(self.cs_y.derivative()(s0), self.cs_x.derivative()(s0))
        dtheta = self.normalize_angle(self.current_yaw - path_yaw)
        # rospy.loginfo("dtheta = %.3f, path_yaw = %.3f", dtheta * 180/math.pi, path_yaw * 180/math.pi)
        q_i = q0
        dq_i = math.tan(dtheta)
        # 클램핑: dq_i의 값이 너무 크면 최대/최소값으로 제한
        # max_dq = 0.1  # 필요에 따라 조정 (예: 1.0 또는 적절한 값)
        # if dq_i > max_dq:
        #     dq_i = max_dq
        # elif dq_i < -max_dq:
        #     dq_i = -max_dq
        # 끝점: 원하는 후보 lane_offset 값을 그대로 사용 (즉, q_f = lane_offset)
        q_f = lane_offset
        dq_f = 0.0

        # 3) 3차 스플라인 계수 (t기준)
        # q(t) = a t^3 + b t^2 + c t + d
        # 경계 조건:
        #   q(0)=d_ = q_i
        #   q'(0)=c_ = dq_i
        #   q(final_delta_s)= a*(Δs)^3 + b*(Δs)^2 + c_*(Δs) + d_ = q_f
        #   q'(final_delta_s)= 3a*(Δs)^2 + 2b*(Δs) + c_ = dq_f
        a_, b_, c_, d_ = self.solve_cubic_spline_coeffs(q_i, dq_i, q_f, dq_f, final_delta_s)

        # 4) t 구간 샘플링
        num_samples = 50
        t_samples = np.linspace(0.0, final_delta_s, num_samples)

        for t in t_samples:
            # 4-1) q(t) 계산
            q_val = self.eval_q_spline_t(a_, b_, c_, d_, t)
            # 4-2) 전역 경로 호 길이 s = (s0 + t) % total_length
            s_val = (s0 + t) % self.total_length
            # print("s_val = %.3f, q_val = %.3f",s_val,q_val)
            # 4-3) Frenet -> Cartesian 변환
            # X, Y = self.frenet_to_cartesian(s_val, q_val)

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = s_val
            pose.pose.position.y = q_val
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        return path_msg


    # ---------------------------------------------------------
    # 3-1) 3차 스플라인 q(s) 계수 계산 (경계 조건 만족)
    # ---------------------------------------------------------
    def solve_cubic_spline_coeffs(self, q_i, dq_i, q_f, dq_f, ds):
        """
        t \in [0, ds] 범위에서,
        q(0)=q_i, q'(0)=dq_i
        q(ds)=q_f, q'(ds)=dq_f
        를 만족하는 a_, b_, c_, d_ 계산
        """
        d_ = q_i       # q(0)=d_
        c_ = dq_i      # q'(0)=c_
        X_f = ds       # 끝점 t=ds

        # 경계조건:
        # q(ds)= a*X_f^3 + b*X_f^2 + c_*X_f + d_ = q_f
        # q'(ds)= 3a*X_f^2 + 2b*X_f + c_ = dq_f

        # 식1: a*ds^3 + b*ds^2 = q_f - (c_*ds + d_)
        # 식2: 3a*ds^2 + 2b*ds = dq_f - c_
        A = np.array([
            [X_f**3, X_f**2],
            [3*X_f**2, 2*X_f]
        ], dtype=float)
        B = np.array([
            [q_f - (c_*X_f + d_)],
            [dq_f - c_]
        ], dtype=float)

        sol = np.linalg.solve(A, B)
        a_ = sol[0,0]
        b_ = sol[1,0]
        return a_, b_, c_, d_

    def eval_q_spline_t(self, a_, b_, c_, d_, t):
        """
        q(t) = a_*t^3 + b_*t^2 + c_*t + d_
        """
        return a_*t**3 + b_*t**2 + c_*t + d_

    # ---------------------------------------------------------
    # 3-2) Frenet -> Cartesian 변환
    # ---------------------------------------------------------
    def frenet_to_cartesian(self, s, q):
        px = self.cs_x(s % self.total_length)
        py = self.cs_y(s % self.total_length)

        tx = self.cs_x.derivative()(s % self.total_length)
        ty = self.cs_y.derivative()(s % self.total_length)
        mag_t = math.hypot(tx, ty)
        if mag_t < 1e-9:
            return px, py
        tx /= mag_t
        ty /= mag_t

        # 법선 벡터: (nx, ny) = (-ty, tx)
        nx = -ty
        ny = tx

        # 올바른 변환: (X, Y) = (px, py) + q * (nx, ny)
        X = px + q * nx
        Y = py + q * ny
        return (X, Y)

    def compute_optimal_path(self,candidate_paths):
        # c_total = w_s * c_s + w_sm * c_sm + w_g + c_g + w_d * c_d
        set_c_s = self.compute_static_obstacle_cost(candidate_paths, len(candidate_paths), self.static_obstacles, threshold=0.25, sigma=1.0)
        pass

    def compute_static_obstacle_cost(self, candidate_paths, number_of_candidate_paths, static_obstacles, threshold=0.25, sigma=1.0):
        indicators = np.zeros(number_of_candidate_paths, dtype = float)
        # 1. 경로 후보의 각 포인트 (x,y)를 추출합니다.
        for pathnum, candidate_path in enumerate(candidate_paths):
            points = []
            for pose in candidate_path.poses:
                s = pose.pose.position.x
                q = pose.pose.position.y
                points.append((s, q))
            points = np.array(points)  # shape: (N, 2)

            # 2. 각 경로 점에서, 모든 정적 장애물과의 거리 계산 후 충돌 여부 판단
            if len(static_obstacles) > 0:
                min_distances = [min(np.hypot(obs[0] - s, obs[1] - q) for obs in static_obstacles) for s, q in points]
                # 임계값보다 작은 값이 하나라도 있으면 1, 아니면 0 (max() 사용)
                indicators[pathnum] = max(1 if d < threshold else 0 for d in min_distances)
            else:
                indicators[pathnum] = 0  # 장애물이 없으면 0 유지

        # 3. 가우시안 필터를 적용하여 비용을 부드럽게 조정
        cost_profile = gaussian_filter1d(indicators, sigma=sigma)

        return cost_profile

if __name__ == '__main__':
    try:
        GlobalPathAnalyzer()
    except rospy.ROSInterruptException:
        pass

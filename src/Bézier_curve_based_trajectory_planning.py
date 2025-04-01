#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import math
import numpy as np
from math import pi, hypot, sqrt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import simps
from scipy.ndimage import gaussian_filter1d

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion
from morai_msgs.msg import EgoVehicleStatus, CandidatePaths, ObjectStatusList
from copy import deepcopy

class GlobalPathAnalyzer:
    def __init__(self):
        rospy.init_node('local_path_pub', anonymous=True)

        # 전역 경로 보간 관련 변수
        self.inside_cs_x = None
        self.inside_cs_y = None
        self.inside_s_vals = None
        self.inside_total_length = None
        self.is_inside_global_path_ready = False
        self.inside_s_candidates = None

        self.outside_cs_x = None
        self.outside_cs_y = None
        self.outside_s_vals = None
        self.outside_total_length = None
        self.is_outside_global_path_ready = False
        self.outside_s_candidates = None

        self.choiced_path = None # 0이면 inside, 1이면 outside
        self.sub_q = None

        self.inside_q0 = 0
        self.outside_q0 = 0
        self.s0 = 0.0

        # 오도메트리 수신 여부 및 위치/헤딩
        self.is_odom_received = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0   # 라디안
        self.current_speed = 0.0

        # 장애물 정보
        self.is_obstacle  =False
        self.obstacles_s = []
        # self.static_obstacles = []

        # 최적화: 최적 제어점을 찾기 위한 비용 함수 최소화 (독립 변수: [s0, x2, s4])
        self.init_params = [2.0, 5.0, 2.0]
        # 프레넷 좌표계에서 현재~목표 사이 거리
        self.s_length_t = 10
        self.target_x = 0.0
        self.target_y = 0.0

        self.kappa0 = 0.0

        self.optimal_params = None
        self.num_points = 100

        self.w = 1.0
        self.offset = 0.0
        self.delta_s = 1.0
        self.delta_d = 0.5
        self.epsilon = 1e-3

        # 디버그 및 로컬 경로 발행
        # self.debug_pub = rospy.Publisher('/debug_s0_q0', Float32MultiArray, queue_size=1)
        self.optimal_path_pub = rospy.Publisher('/local_path', Path, queue_size=1)

        # 토픽 구독
        rospy.Subscriber("/inside_global_path", Path, self.inside_global_path_callback)
        rospy.Subscriber("/outside_global_path", Path, self.outside_global_path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        rospy.Subscriber("/Object_topic", ObjectStatusList, self.obstacle_callback)
        rospy.Subscriber("Ego_topic", EgoVehicleStatus, self.status_callback)

        rate = rospy.Rate(20)  # 20Hz
        while not rospy.is_shutdown():
            #os.system('clear')
            if self.is_odom_received and self.is_inside_global_path_ready and self.is_outside_global_path_ready and self.is_status and self.is_obstacle:
                """
                주어진 전역 좌표 (x, y)에 대해, 전역 경로 상의 s 값을 벡터화된 방법으로 계산합니다.
                """
                # 만약 self.s_candidates가 존재하지 않으면 미리 계산해둡니다.
                if not hasattr(self, 'inside_s_candidates') or self.inside_s_candidates is None:
                    num_samples = 3000  # 필요에 따라 조절
                    self.inside_s_candidates = np.linspace(0, self.inside_total_length, num_samples)
                if not hasattr(self, 'outside_s_candidates') or self.outside_s_candidates is None:
                    num_samples = 3000  # 필요에 따라 조절
                    self.outside_s_candidates = np.linspace(0, self.outside_total_length, num_samples)

                # 1) 현재 위치로부터 s0 계산
                inside_s0 = self.inside_find_closest_s_newton(self.current_x, self.current_y)
                outside_s0 = self.outside_find_closest_s_newton(self.current_x, self.current_y)
                # 2) 현재 횡방향 오프셋 q0 계산
                self.inside_q0 = self.inside_signed_lateral_offset(self.current_x, self.current_y, inside_s0)
                self.outside_q0 = self.outside_signed_lateral_offset(self.current_x, self.current_y, outside_s0)

                print("inside_q0:",self.inside_q0,'outside_q0:',self.outside_q0)
                self.sub_q = abs(self.inside_q0 - self.outside_q0)
                print("sub_q:",self.sub_q)
                if abs(self.inside_q0) <= abs(self.outside_q0):
                    self.s0 = inside_s0
                    q0 = self.inside_q0
                    self.choiced_path = 0
                else:
                    self.s0 = outside_s0
                    q0 = self.outside_q0
                    self.choiced_path = 1

                print(self.choiced_path)

                self.kappa0 = self.global_curvature(self.s0)

                self.target_x, self.target_y = self.frenet_to_cartesian(self.s0 + self.s_length_t, 0)
                if self.choiced_path == 0:
                    self.target_yaw = math.atan2(self.inside_cs_y.derivative()((self.s0 + self.s_length_t) % self.inside_total_length), self.inside_cs_x.derivative()((self.s0 + self.s_length_t) % self.inside_total_length))
                else:
                    self.target_yaw = math.atan2(self.outside_cs_y.derivative()((self.s0 + self.s_length_t) % self.outside_total_length), self.outside_cs_x.derivative()((self.s0 + self.s_length_t) % self.outside_total_length))

                print("kappa0, current_yaw, target_yaw:", self.kappa0, self.current_yaw * 180/3.14, self.target_yaw * 180/3.14)

                # self.optimal_params = minimize(self.cost_function, self.init_params,
                #    args=(),
                #    bounds=[(0.1, 10), (0.1, 10), (0.1, 10)],
                #    method='SLSQP')
                # print("Optimized parameters [s0, x2, s4]:", self.optimal_params)
                self.optimal_control_points = self.generate_control_points(self.init_params)
                print("Optimal Control Points:", self.optimal_control_points)
                taus = np.linspace(0, 1, self.num_points)
                self.optimal_curve = np.array([self.bezier_curve(self.optimal_control_points, tau) for tau in taus])
                # 최적 Bézier 곡선의 호 길이 계산
                self.s_T = self.bezier_arc_length(self.optimal_control_points)
                print("Arc length of optimal Bézier curve (s_T):", self.s_T)

                self.publish_optimal_path(self.optimal_curve)

                self.is_odom_received = False
                self.is_status = False
                self.is_obstacle = False

            rate.sleep()


    def status_callback(self, msg):
        self.is_status = True
        self.current_speed = msg.velocity.x * 3.75

    # ---------------------------------------------------------
    # 전역 경로 /global_path 콜백
    # ---------------------------------------------------------
    def inside_global_path_callback(self, msg):
        if not msg.poses:
            rospy.logwarn("수신한 inside_global_path가 비어 있습니다.")
            return

        if self.is_inside_global_path_ready is False:
            x_points = []
            y_points = []
            for pose_stamped in msg.poses:
                x_points.append(pose_stamped.pose.position.x)
                y_points.append(pose_stamped.pose.position.y)

            # 폐곡선 가정: 첫번째와 마지막 좌표가 같도록 보정
            tol = 1e-8
            if abs(x_points[0] - x_points[-1]) > tol or abs(y_points[0] - y_points[-1]) > tol:
                rospy.logwarn("inside_global_path 첫번째와 마지막 좌표가 다릅니다. 강제로 보정합니다.")
                x_points[-1] = x_points[0]
                y_points[-1] = y_points[0]

            # 호 길이 s_vals 계산
            s_vals = [0.0]
            for i in range(1, len(x_points)):
                dist = hypot(x_points[i] - x_points[i-1],
                            y_points[i] - y_points[i-1])
                s_vals.append(s_vals[-1] + dist)

            self.inside_s_vals = s_vals
            self.inside_total_length = s_vals[-1]

            # 주기적 스플라인 보간
            self.inside_cs_x = CubicSpline(self.inside_s_vals, x_points, bc_type='periodic')
            self.inside_cs_y = CubicSpline(self.inside_s_vals, y_points, bc_type='periodic')

            self.is_inside_global_path_ready = True
            # rospy.loginfo("글로벌 경로 보간 완료. 전체 길이=%.3f", self.total_length)

    def outside_global_path_callback(self, msg):
        if not msg.poses:
            rospy.logwarn("수신한 global_path가 비어 있습니다.")
            return

        if self.is_inside_global_path_ready is False:
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

            self.outside_s_vals = s_vals
            self.outside_total_length = s_vals[-1]

            # 주기적 스플라인 보간
            self.outside_cs_x = CubicSpline(self.outside_s_vals, x_points, bc_type='periodic')
            self.outside_cs_y = CubicSpline(self.outside_s_vals, y_points, bc_type='periodic')

            self.is_outside_global_path_ready = True

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
        ObjectStatusList 메시지에서 obstacle_list에 있는 각 동적 장애물의 위치, 속도, 가속도를 이용하여,
        전역 경로 상의 Frenet 좌표 (s, q)와 함께 속도, 가속도 (v_s, v_q, a_s, a_q)를 계산하고,
        self.dynamic_obstacles에 저장합니다.
        각 장애물은 (s, q, v_s, v_q, a_s, a_q) 튜플로 저장됩니다.
        """
        if not self.is_inside_global_path_ready and not self.is_outside_global_path_ready:
            rospy.logwarn("전역 경로가 아직 준비되지 않았습니다.")
            return
        self.is_obstacle = True
        dyn_obs = []
        for obstacle in msg.obstacle_list:
            # 장애물의 전역 위치 (x, y)
            x_obs = obstacle.position.x
            y_obs = obstacle.position.y
            # 전역 속도 벡터 (v_x, v_y)와 가속도 벡터 (a_x, a_y)
            v_x = obstacle.velocity.x
            v_y = obstacle.velocity.y
            a_x = obstacle.acceleration.x
            a_y = obstacle.acceleration.y
            # Frenet 좌표 및 속도/가속도 변환
            s_obs, q_obs = self.compute_obstacle_frenet_all(x_obs, y_obs, v_x, v_y, a_x, a_y)
            # |q_obs|가 1.5 이하인 경우만 저장
            if abs(q_obs) <= 3.0:
                dyn_obs.append((s_obs, q_obs, v_x))
        self.obstacles_s = dyn_obs
        # rospy.loginfo("동적 장애물 업데이트: %s", self.obstacles_s)

    def compute_obstacle_frenet_all(self, x_obs, y_obs, v_x, v_y, a_x, a_y):
        """
        주어진 전역 좌표 (x_obs, y_obs), 속도 (v_x, v_y), 가속도 (a_x, a_y)를
        전역 경로에 투영하여, Frenet 좌표 (s, q)와 함께, 속도와 가속도를 Frenet 좌표 성분(v_s, v_q, a_s, a_q)으로 변환합니다.
        """
        # 1. 위치 변환: s, q 계산 (기존 방식)
        s_obs = self.compute_s_coordinate(x_obs, y_obs)
        if self.choiced_path == 0:
            q_obs = self.inside_signed_lateral_offset(x_obs, y_obs, s_obs)
            # T_x = self.inside_cs_x.derivative()(s_obs)
            # T_y = self.inside_cs_y.derivative()(s_obs)
        else:
            q_obs = self.outside_signed_lateral_offset(x_obs, y_obs, s_obs)
            # T_x = self.outside_cs_x.derivative()(s_obs)
            # T_y = self.outside_cs_y.derivative()(s_obs)

        # 2. 전역 경로에서 s_obs 위치에서 접선 벡터 T와 법선 벡터 N 계산
        # T = (dx/ds, dy/ds) normalized
        # T_norm = sqrt(T_x**2 + T_y**2)
        # if T_norm < 1e-6:
        #     T_norm = 1e-6
        # T_x /= T_norm
        # T_y /= T_norm

        # N = (-T_y, T_x) (여기서는 왼쪽이 +q라고 가정)
        # N_x = -T_y
        # N_y = T_x

        # 3. 전역 속도와 가속도를 Frenet 성분으로 변환
        # 속도: v_s = v · T, v_q = v · N
        # v_s = v_x * T_x + v_y * T_y
        # v_q = v_x * N_x + v_y * N_y

        # 4. 전역 가속도 벡터를 Frenet 성분으로 변환
        # 곡률 kappa = (x'(s)*y''(s) - y'(s)*x''(s)) / ((x'(s)^2 + y'(s)^2)^(3/2))
        # if self.choiced_path == 0:
        #     cs_x_prime = self.inside_cs_x.derivative()(s_obs)
        #     cs_y_prime = self.inside_cs_y.derivative()(s_obs)
        #     cs_x_second = self.inside_cs_x.derivative(2)(s_obs)
        #     cs_y_second = self.inside_cs_y.derivative(2)(s_obs)
        # else:
        #     cs_x_prime = self.outside_cs_x.derivative()(s_obs)
        #     cs_y_prime = self.outside_cs_y.derivative()(s_obs)
        #     cs_x_second = self.outside_cs_x.derivative(2)(s_obs)
        #     cs_y_second = self.outside_cs_y.derivative(2)(s_obs)
        # denom = (cs_x_prime**2 + cs_y_prime**2)**1.5
        # if denom < 1e-6:
        #     kappa = 0.0
        # else:
        #     kappa = (cs_x_prime * cs_y_second - cs_y_prime * cs_x_second) / denom

        # Frenet 가속도 공식:
        # a_s = T·a - kappa * v_q^2
        # a_q = N·a + kappa * v_s * v_q
        # a_s = a_x * T_x + a_y * T_y - kappa * (v_q**2)
        # a_q = a_x * N_x + a_y * N_y + kappa * v_s * v_q

        return s_obs, q_obs

    def compute_s_coordinate(self, x, y):
        """
        주어진 전역 좌표 (x, y)에 대해, 전역 경로 상의 s 값을 벡터화된 방법으로 계산합니다.
        self.s_candidates는 미리 생성되어 있다고 가정합니다.
        """
        if self.choiced_path == 0:
            xs = self.inside_cs_x(self.inside_s_candidates)
            ys = self.inside_cs_y(self.inside_s_candidates)
            distances = (x - xs)**2 + (y - ys)**2
            s_best = self.inside_s_candidates[np.argmin(distances)]
        else:
            xs = self.outside_cs_x(self.outside_s_candidates)
            ys = self.outside_cs_y(self.outside_s_candidates)
            distances = (x - xs)**2 + (y - ys)**2
            s_best = self.outside_s_candidates[np.argmin(distances)]
        return s_best

    # ---------------------------------------------------------
    # 1) inside path s0 찾기 (2차 최소화와 뉴튼 방법)
    # ---------------------------------------------------------
    def inside_find_closest_s_newton(self, x0, y0):
        if not self.is_inside_global_path_ready:
            return 0.0

        # if self.inside_prev_s is not None:
        #     s_current = self.inside_prev_s
        # else:
        #     # CubicSpline은 벡터화되어 있어, 한 번에 여러 s 값을 평가할 수 있음
        #     xs = self.inside_cs_x(self.inside_s_candidates)
        #     ys = self.inside_cs_y(self.inside_s_candidates)
        #     distances = (x0 - xs)**2 + (y0 - ys)**2  # 벡터화된 거리 계산
        #     s_current = self.inside_s_candidates[np.argmin(distances)]

        xs = self.inside_cs_x(self.inside_s_candidates)
        ys = self.inside_cs_y(self.inside_s_candidates)
        distances = (x0 - xs)**2 + (y0 - ys)**2  # 벡터화된 거리 계산
        s_current = self.inside_s_candidates[np.argmin(distances)]

        max_iter = 30
        tol = 1e-6
        for _ in range(max_iter):
            fprime = self.inside_dist_sq_grad(s_current, x0, y0)
            fsecond = self.inside_dist_sq_hess(s_current, x0, y0)
            if abs(fsecond) < 1e-12:
                break
            step = -fprime / fsecond
            s_current += step
            s_current = s_current % self.inside_total_length
            if abs(step) < tol:
                break

        return s_current

    def inside_dist_sq_grad(self, s, x0, y0):
        dx = x0 - self.inside_cs_x(s)
        dy = y0 - self.inside_cs_y(s)
        dxds = self.inside_cs_x.derivative()(s)
        dyds = self.inside_cs_y.derivative()(s)
        return -2.0 * (dx * dxds + dy * dyds)

    def inside_dist_sq_hess(self, s, x0, y0):
        dx = x0 - self.inside_cs_x(s)
        dy = y0 - self.inside_cs_y(s)
        dxds = self.inside_cs_x.derivative()(s)
        dyds = self.inside_cs_y.derivative()(s)
        d2xds2 = self.inside_cs_x.derivative(2)(s)
        d2yds2 = self.inside_cs_y.derivative(2)(s)
        val = (-dxds**2 + dx * d2xds2) + (-dyds**2 + dy * d2yds2)
        return -2.0 * val

    # ---------------------------------------------------------
    # 2) 현재 횡방향 오프셋 q0
    # ---------------------------------------------------------
    def inside_signed_lateral_offset(self, x0, y0, s0):
        px = self.inside_cs_x(s0)
        py = self.inside_cs_y(s0)
        tx = self.inside_cs_x.derivative()(s0)
        ty = self.inside_cs_y.derivative()(s0)
        dx_veh = x0 - px
        dy_veh = y0 - py
        cross_val = tx * dy_veh - ty * dx_veh
        q0 = math.sqrt(dx_veh**2 + dy_veh**2)
        return +q0 if cross_val > 0.0 else -q0

    # ---------------------------------------------------------
    # 1) outside path s0 찾기 (2차 최소화와 뉴튼 방법)
    # ---------------------------------------------------------

    def outside_find_closest_s_newton(self, x0, y0):
        if not self.is_outside_global_path_ready:
            return 0.0

        # if self.outside_prev_s is not None:
        #     s_current = self.outside_prev_s
        # else:
        #     # CubicSpline은 벡터화되어 있어, 한 번에 여러 s 값을 평가할 수 있음
        #     xs = self.outside_cs_x(self.outside_s_candidates)
        #     ys = self.outside_cs_y(self.outside_s_candidates)
        #     distances = (x0 - xs)**2 + (y0 - ys)**2  # 벡터화된 거리 계산
        #     s_current = self.outside_s_candidates[np.argmin(distances)]
        xs = self.outside_cs_x(self.outside_s_candidates)
        ys = self.outside_cs_y(self.outside_s_candidates)
        distances = (x0 - xs)**2 + (y0 - ys)**2  # 벡터화된 거리 계산
        s_current = self.outside_s_candidates[np.argmin(distances)]

        max_iter = 30
        tol = 1e-6
        for _ in range(max_iter):
            fprime = self.outside_dist_sq_grad(s_current, x0, y0)
            fsecond = self.outside_dist_sq_hess(s_current, x0, y0)
            if abs(fsecond) < 1e-12:
                break
            step = -fprime / fsecond
            s_current += step
            s_current = s_current % self.outside_total_length
            if abs(step) < tol:
                break

        return s_current

    def outside_dist_sq_grad(self, s, x0, y0):
        dx = x0 - self.outside_cs_x(s)
        dy = y0 - self.outside_cs_y(s)
        dxds = self.outside_cs_x.derivative()(s)
        dyds = self.outside_cs_y.derivative()(s)
        return -2.0 * (dx * dxds + dy * dyds)

    def outside_dist_sq_hess(self, s, x0, y0):
        dx = x0 - self.outside_cs_x(s)
        dy = y0 - self.outside_cs_y(s)
        dxds = self.outside_cs_x.derivative()(s)
        dyds = self.outside_cs_y.derivative()(s)
        d2xds2 = self.outside_cs_x.derivative(2)(s)
        d2yds2 = self.outside_cs_y.derivative(2)(s)
        val = (-dxds**2 + dx * d2xds2) + (-dyds**2 + dy * d2yds2)
        return -2.0 * val

    # ---------------------------------------------------------
    # 2) 현재 횡방향 오프셋 q0
    # ---------------------------------------------------------
    def outside_signed_lateral_offset(self, x0, y0, s0):
        px = self.outside_cs_x(s0)
        py = self.outside_cs_y(s0)
        tx = self.outside_cs_x.derivative()(s0)
        ty = self.outside_cs_y.derivative()(s0)
        dx_veh = x0 - px
        dy_veh = y0 - py
        cross_val = tx * dy_veh - ty * dx_veh
        q0 = math.sqrt(dx_veh**2 + dy_veh**2)
        return +q0 if cross_val > 0.0 else -q0

    # ---------------------------------------------------------
    # 3-2) Frenet -> Cartesian 변환
    # ---------------------------------------------------------
    def frenet_to_cartesian(self, s, q):
        if self.choiced_path == 0:
            px = self.inside_cs_x(s % self.inside_total_length)
            py = self.inside_cs_y(s % self.inside_total_length)

            tx = self.inside_cs_x.derivative()(s % self.inside_total_length)
            ty = self.inside_cs_y.derivative()(s % self.inside_total_length)
        else:
            px = self.outside_cs_x(s % self.outside_total_length)
            py = self.outside_cs_y(s % self.outside_total_length)

            tx = self.outside_cs_x.derivative()(s % self.outside_total_length)
            ty = self.outside_cs_y.derivative()(s % self.outside_total_length)
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

    def global_curvature(self, s):
        if self.choiced_path == 0:
            dx_ds = self.inside_cs_x.derivative()(s)
            dy_ds = self.inside_cs_y.derivative()(s)
            d2x_ds2 = self.inside_cs_x.derivative(2)(s)
            d2y_ds2 = self.inside_cs_y.derivative(2)(s)
        else:
            dx_ds = self.outside_cs_x.derivative()(s)
            dy_ds = self.outside_cs_y.derivative()(s)
            d2x_ds2 = self.outside_cs_x.derivative(2)(s)
            d2y_ds2 = self.outside_cs_y.derivative(2)(s)
        numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
        denominator = (dx_ds**2 + dy_ds**2)**1.5
        return 0.0 if abs(denominator) < 1e-6 else numerator / denominator

    def bernstein_poly(self, i, n, tau):
        return math.comb(n, i) * (tau**i) * ((1 - tau)**(n - i))

    def bezier_curve(self, control_points, tau):
        n = 4
        point = np.zeros(2)
        for i in range(n + 1):
            point += self.bernstein_poly(i, n, tau) * np.array(control_points[i])
        return point

    def bezier_derivative(self, control_points, tau):
        n = 4
        d_point = np.zeros(2)
        for i in range(n):
            d_point += math.comb(n - 1, i) * (tau**i) * ((1 - tau)**(n - 1 - i)) * (control_points[i + 1] - control_points[i]) * n
        return d_point

    def bezier_second_derivative(self, control_points, tau):
        n = 4
        dd_point = np.zeros(2)
        for i in range(n - 1):
            dd_point += math.comb(n - 2, i) * (tau**i) * ((1 - tau)**(n - 2 - i)) * (control_points[i + 2] - 2 * control_points[i + 1] + control_points[i]) * (n * (n - 1))
        return dd_point

    def curvature(self, control_points, tau):
        d1 = self.bezier_derivative(control_points, tau)
        d2 = self.bezier_second_derivative(control_points, tau)
        num = d1[0] * d2[1] - d1[1] * d2[0]
        denom = (d1[0]**2 + d1[1]**2)**1.5
        return 0.0 if abs(denom) < 1e-6 else num / denom

    def bezier_arc_length(self, control_points):
        taus = np.linspace(0, 1, self.num_points)
        pts = np.array([self.bezier_curve(control_points, tau) for tau in taus])
        length = 0.0
        for i in range(1, len(pts)):
            length += np.linalg.norm(pts[i] - pts[i - 1])
        return length

    def map_d_to_angle(self, d):
        if self.choiced_path == 0:
            if (self.inside_q0 > 0 and self.outside_q0 > 0 and self.inside_q0 < self.outside_q0) or (self.inside_q0 <= 0 and self.outside_q0 >= 0) or (self.inside_q0 < 0 and self.outside_q0 < 0 and self.inside_q0 < self.outside_q0):
               pass
            else:
                d = -d
            if d >= 0:
                return ((self.w - d) / self.w) * math.pi
            elif d >= -self.sub_q:
                return math.pi
            else:
                return math.pi + ((d + self.sub_q) / self.w) * math.pi
        else:
            if (self.inside_q0 > 0 and self.outside_q0 > 0 and self.inside_q0 < self.outside_q0) or (self.inside_q0 <= 0 and self.outside_q0 >= 0) or (self.inside_q0 < 0 and self.outside_q0 < 0 and self.inside_q0 < self.outside_q0):
               d = -d
            else:
                pass
            if d >= 0:
                return ((self.w - d) / self.w) * math.pi
            elif d >= -self.sub_q:
                return math.pi
            else:
                return math.pi + ((d + self.sub_q) / self.w) * math.pi

    def compute_U_road_modified(self, d):
        x_val = self.map_d_to_angle(d)
        return 0.5 * (math.cos(x_val) + 1)

    def compute_U_obs(self, s, d, s_i, d_i, v_i):
        if (self.current_speed >= v_i and self.s0 <= s_i) or (self.current_speed < v_i and self.s0 > s_i):
            rel_speed_term = 1.0 / abs(v_e - v_i) + epsilon
            exp_term = - rel_speed_term * (((s - s_i)**2)/(delta_s**2) + ((d - d_i)**2)/(delta_d**2))
            return np.exp(exp_term)
        else:
            return 0.0

    def compute_total_potential_field(self, s, d):
        U_road_val = self.compute_U_road_modified(d)
        U_obs_total = 0.0
        for obs in self.obstacles_s:
            s_i, d_i, v_i = obs
            U_obs_total += self.compute_U_obs(s, d, s_i, d_i, v_i)
        return max(U_road_val, U_obs_total)

    def U_total_from_field(self, control_points, tau):
        point = self.bezier_curve(control_points, tau)
        x, y = point[0], point[1]
        if self.choiced_path == 0:
            s_val = self.inside_find_closest_s_newton(x, y)
            d_val = self.inside_signed_lateral_offset(x, y, s_val)
        else:
            s_val = self.outside_find_closest_s_newton(x, y)
            d_val = self.outside_signed_lateral_offset(x, y, s_val)
        return self.compute_total_potential_field(s_val, d_val)

    def cost_function(self, params):
        control_points = self.generate_control_points(params)
        taus = np.linspace(0, 1, self.num_points)
        curvatures = np.array([self.curvature(control_points, tau) for tau in taus])
        U_vals = np.array([self.U_total_from_field(control_points, tau) for tau in taus])
        J_curv = simps(curvatures**2, taus)
        J_U = simps(U_vals, taus)
        return J_curv + J_U

    def generate_control_points(self, params):
        """
        새로운 좌표계에서 제어점을 생성한 후 전역 좌표계로 변환합니다.

        입력:
        params: [s0, x2, s4] (독립 변수)
        self.current_x, self.current_y: 차량의 현재 전역 위치
        self.target_x, self.target_y: 목표 전역 위치
        self.current_yaw, self.target_yaw: 현재 및 목표 헤딩 (라디안)
        self.kappa0: 초기 곡률
        출력:
        5개의 제어점 [P0, P1, P2, P3, P4] (각각 전역 좌표계 상의 [x, y])
        """
        s0, x2, s4 = params
        # 로컬 좌표계에서 P0는 (0,0)
        P0 = np.array([0.0, 0.0])
        P1 = np.array([s0, 0.0])
        # P2: y 성분은 (4 * kappa0 * s0^2) / 3, x 성분은 자유 변수 x2
        P2 = np.array([x2, (4 * self.kappa0 * (s0**2)) / 3.0])

        # 전역 목표 상태를 로컬 좌표계로 변환 (회전행렬 이용)
        # T = np.array([[np.cos(self.current_yaw), -np.sin(self.current_yaw)],
        #             [np.sin(self.current_yaw),  np.cos(self.current_yaw)]])
        # p4_input = np.array([self.target_x - self.current_x, self.target_y - self.current_y])
        # P4_local = T.dot(p4_input)
        # P4 = np.array([P4_local[0], P4_local[1]])

        # P3: 목표 상태 P4에서 s4만큼 뒤로 이동 (목표 헤딩 theta_t를 반영하여)
        # P3 = np.array([P4[0] - s4 * np.cos(self.target_yaw - self.current_yaw),
        #             P4[1] - s4 * np.sin(self.target_yaw - self.current_yaw)])

        global_P4 = np.array([self.target_x, self.target_y])
        local_P4 = np.array(self.to_local(global_P4[0], global_P4[1]))
        P3 = np.array([local_P4[0] - s4 * math.cos(self.target_yaw - self.current_yaw), local_P4[1] - s4 * math.sin(self.target_yaw - self.current_yaw)])

        # 로컬 좌표계에서 계산된 제어점들을 전역 좌표계로 변환
        global_P0 = np.array(self.to_global(P0[0], P0[1]))
        global_P1 = np.array(self.to_global(P1[0], P1[1]))
        global_P2 = np.array(self.to_global(P2[0], P2[1]))
        global_P3 = np.array(self.to_global(P3[0], P3[1]))
        # global_P4 = np.array(self.to_global(P4[0], P4[1]))

        return [global_P0, global_P1, global_P2, global_P3, global_P4]

    def publish_optimal_path(self, curve):
        """
        최적 경로 (numpy array, 전역 좌표계)를 nav_msgs/Path 메시지로 변환하여 발행합니다.
        """
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        for pt in curve:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "map"
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.orientation.w = 1.0  # 간단하게 설정
            path_msg.poses.append(pose)
        self.optimal_path_pub.publish(path_msg)
        rospy.loginfo("Optimal path published with %d poses.", len(path_msg.poses))

    def to_global(self, x, y):
        T = np.array([
            [np.cos(self.current_yaw), -np.sin(self.current_yaw), self.current_x],
            [np.sin(self.current_yaw),  np.cos(self.current_yaw), self.current_y],
                                              [0,              0,              1]])
        local_point = np.array([x, y, 1])
        global_point = T.dot(local_point)
        return global_point[0], global_point[1]

    def to_local(self, x, y):
        T = np.array([
            [np.cos(self.current_yaw), -np.sin(self.current_yaw), self.current_x],
            [np.sin(self.current_yaw),  np.cos(self.current_yaw), self.current_y],
                                              [0,              0,              1]])
        T_inv = np.linalg.inv(T)
        global_point = np.array([x, y, 1])
        local_point = T_inv.dot(global_point)
        return local_point[0], local_point[1]


if __name__ == '__main__':
    try:
        GlobalPathAnalyzer()
    except rospy.ROSInterruptException:
        pass

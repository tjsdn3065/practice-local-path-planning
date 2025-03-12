#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import math
import numpy as np
from math import pi, hypot, sqrt
from scipy.interpolate import CubicSpline
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
        self.optimal_path_pub = rospy.Publisher('/local_path', Path, queue_size=1)

        # 토픽 구독
        rospy.Subscriber("/inside_global_path", Path, self.inside_global_path_callback)
        rospy.Subscriber("/outside_global_path", Path, self.outside_global_path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        rospy.Subscriber("/Object_topic", ObjectStatusList, self.obstacle_callback)
        rospy.Subscriber("Ego_topic", EgoVehicleStatus, self.status_callback)

        rate = rospy.Rate(20)  # 20Hz
        while not rospy.is_shutdown():
            os.system('clear')
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
                inside_q0 = self.inside_signed_lateral_offset(self.current_x, self.current_y, inside_s0)
                outside_q0 = self.outside_signed_lateral_offset(self.current_x, self.current_y, outside_s0)

                print("inside_q0:",inside_q0,'outside_q0:',outside_q0)
                self.sub_q = abs(inside_q0 - outside_q0)
                print("sub_q:",self.sub_q)
                if abs(inside_q0) <= abs(outside_q0):
                    s0 = inside_s0
                    q0 = inside_q0
                    self.choiced_path = 0
                else:
                    s0 = outside_s0
                    q0 = outside_q0
                    self.choiced_path = 1

                print(self.choiced_path)

                # 3) 로컬 경로 생성 (Δs는 차량 속도 및 장애물 정보 반영)
                # 후보 경로들을 생성 (-1.5부터 1.5까지 0.1 단위)
                candidate_paths_list, cart_candidate_paths_list = self.generate_cadidate_paths(s0, q0)

                # CandidatePaths 커스텀 메시지에 모든 후보 경로 저장
                candidate_paths_msg = CandidatePaths()
                candidate_paths_msg.paths = candidate_paths_list

                cart_candidate_paths_msg = CandidatePaths()
                cart_candidate_paths_msg.paths = cart_candidate_paths_list

                # 최적 경로 선택 후
                optimal_path, min_idx = self.compute_optimal_path(candidate_paths_msg.paths)
                self.optimal_path_pub.publish(optimal_path)

                # cart_candidate_paths_msg.paths 는 후보 경로들이 Cartesian 좌표로 변환된 리스트라고 가정합니다.
                # 각 후보 경로를 발행하는 publisher들을 리스트로 만듭니다.
                all_publishers = [
                    self.candidate_path_pub1,
                    self.candidate_path_pub2,
                    self.candidate_path_pub3,
                    self.candidate_path_pub4,
                    self.candidate_path_pub5,
                    self.candidate_path_pub6,
                    self.candidate_path_pub7
                ]

                # 후보 경로 리스트와 해당 인덱스를 추출합니다.
                candidate_paths = cart_candidate_paths_msg.paths  # 예: 이미 Cartesian 좌표로 변환된 후보 경로 리스트
                # 최적 경로(인덱스 min_idx)를 제외한 후보 경로와 그에 대응하는 publisher 리스트를 생성합니다.
                filtered_paths = [path for i, path in enumerate(candidate_paths) if i != min_idx]
                filtered_publishers = [pub for i, pub in enumerate(all_publishers) if i != min_idx]

                # 각 후보 경로를 대응하는 publisher를 통해 발행합니다.
                for pub, path in zip(filtered_publishers, filtered_paths):
                    pub.publish(path)

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
            # s_obs, q_obs, v_s, v_q, a_s, a_q = self.compute_obstacle_frenet_all(x_obs, y_obs, v_x, v_y, a_x, a_y)
            s_obs, q_obs = self.compute_obstacle_frenet_all(x_obs, y_obs, v_x, v_y, a_x, a_y)
            # |q_obs|가 1.5 이하인 경우만 저장
            if abs(q_obs) <= 1.5:
                dyn_obs.append((s_obs, q_obs))
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

        return s_obs, q_obs#, v_s, v_q, a_s, a_q

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
    # (C) Δs 결정: 속도 및 장애물 정보를 반영 (식 (6) 및 Algorithm 2)
    # ---------------------------------------------------------
    def compute_delta_s_vel(self, v, a_min=-3.0, s_min=5.0, s_max=10.0):
        """
        논문 식 (6)에 따라 차량 속도 v (m/s), 최소 가속도 a_min,
        최소/최대 호 길이(s_min, s_max)를 고려해 Δs를 계산.
        """
        s_candidate = s_min + (v**2 / abs(a_min))
        if s_candidate < s_max:
            return s_candidate
        else:
            return s_max

    def compute_delta_s_with_obstacles(self, s_vehicle, s_vel, s_min=5.0):
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
            return max(s_obs, s_min)

    def normalize_angle(self, angle):
        """
        각도를 -pi에서 pi 사이로 정규화합니다.
        입력 각도는 radian 단위여야 합니다.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # ---------------------------------------------------------
    # 3) 로컬 경로(후보 경로) 생성 (3차 스플라인 q(s) 이용)
    # ---------------------------------------------------------
    def generate_cadidate_paths(self, s0, q0):
        """
        s0: 시작 호 길이 (현재 차량 위치에 해당)
        q0: 현재 횡방향 오프셋 (signed_lateral_offset 함수로 구한 값)
        이 함수를 통해 -1.5부터 1.5까지 0.1 간격의 후보 경로들을 생성합니다.
        """
        candidate_paths = []
        cart_candidate_paths = []
        if self.sub_q <= 0.3:
            for lane_offset in np.arange(-1.0, 1.0 + 0.001, 0.4):
                path_msg, cart_path_msg = self.generate_local_path(s0, q0, lane_offset)
                candidate_paths.append(path_msg)
                cart_candidate_paths.append(cart_path_msg)
        else:
            if self.choiced_path == 0:
                for lane_offset in np.arange(-2.0, 0 + 0.001, 0.4):
                    path_msg, cart_path_msg = self.generate_local_path(s0, q0, lane_offset)
                    candidate_paths.append(path_msg)
                    cart_candidate_paths.append(cart_path_msg)
            else:
                for lane_offset in np.arange(0, 2.0 + 0.001, 0.4):
                    path_msg, cart_path_msg = self.generate_local_path(s0, q0, lane_offset)
                    candidate_paths.append(path_msg)
                    cart_candidate_paths.append(cart_path_msg)
        return candidate_paths, cart_candidate_paths

    def generate_local_path(self, s0, q0, lane_offset):
        """
        s0: 시작 호 길이 (현재 차량 위치에 해당하는 s)
        q0: 현재 횡방향 오프셋 (signed_lateral_offset 함수로 구한 값)
        lane_offset: 후보로 사용할 최종 오프셋 (q_f) 값 (예: -1.5, -1.4, …, 1.5)
        """
        path_msg = Path()
        path_msg.header.frame_id = "map"

        cart_path_msg = Path()
        cart_path_msg.header.frame_id = "map"

        # (a) Δs 결정: 차량 속도와 장애물 정보를 반영하여 Δs를 구함
        s_vel = self.compute_delta_s_vel(self.current_speed, a_min = -3.0, s_min=5.0)
        final_delta_s = self.compute_delta_s_with_obstacles(s0, s_vel, s_min=5.0)

        # (b) 경계 조건:
        # 시작점: q(s0)=q0, q'(s0)=tan(Δθ)
        if self.choiced_path == 0:
            path_yaw = math.atan2(self.inside_cs_y.derivative()(s0), self.inside_cs_x.derivative()(s0))
        else:
            path_yaw = math.atan2(self.outside_cs_y.derivative()(s0), self.outside_cs_x.derivative()(s0))
        dtheta = self.normalize_angle(self.current_yaw - path_yaw)
        # rospy.loginfo("dtheta = %.3f, path_yaw = %.3f", dtheta * 180/math.pi, path_yaw * 180/math.pi)
        q_i = q0
        dq_i = math.tan(dtheta)
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
        num_samples = 10
        t_samples = np.linspace(0.0, final_delta_s, num_samples)

        for t in t_samples:
            # 4-1) q(t) 계산
            q_val = self.eval_q_spline_t(a_, b_, c_, d_, t)
            # 4-2) 전역 경로 호 길이 s = (s0 + t) % total_length
            if self.choiced_path == 0:
                s_val = (s0 + t) % self.inside_total_length
            else:
                s_val = (s0 + t) % self.outside_total_length
            # print("s_val = %.3f, q_val = %.3f",s_val,q_val)
            # 4-3) Frenet -> Cartesian 변환
            X, Y = self.frenet_to_cartesian(s_val, q_val)

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = s_val
            pose.pose.position.y = q_val
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

            cart_pose = PoseStamped()
            cart_pose.header.frame_id = "map"
            cart_pose.pose.position.x = X
            cart_pose.pose.position.y = Y
            cart_pose.pose.orientation.w = 1.0
            cart_path_msg.poses.append(cart_pose)

        return path_msg,cart_path_msg


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

    def convert_frenet_path_to_cartesian(self, frenet_path):
        """
        frenet_path: nav_msgs/Path
        - pose.position.x = s
        - pose.position.y = q
        를 전역 좌표 (X, Y)로 변환한 새 Path를 리턴.
        """
        cartesian_path = deepcopy(frenet_path)
        cartesian_path.poses = []  # Pose 배열은 새로 만든다.

        for pose in frenet_path.poses:
            s_val = pose.pose.position.x
            q_val = pose.pose.position.y
            X, Y = self.frenet_to_cartesian(s_val, q_val)

            new_pose = deepcopy(pose)
            new_pose.pose.position.x = X
            new_pose.pose.position.y = Y
            # 필요하면 orientation도 계산할 수 있으나, 여기서는 w=1.0 등 간단히 처리 가능
            cartesian_path.poses.append(new_pose)

        return cartesian_path

    def compute_optimal_path(self, candidate_paths):
        """
        후보 경로들에 대해
            c_total = w_s*c_s + w_sm*c_sm + w_g*c_g (+ w_d*c_d)
        를 계산하고, 가장 비용이 낮은 경로(최적 경로)를 리턴한다.

        Returns:
            best_path: 비용이 가장 낮은 nav_msgs/Path
            best_cost: 해당 경로의 총 비용
        """
        # 가중치 (동적 장애물 비용 w_d는 0으로 가정)
        w_s = 3.0   # 정적 장애물 비용 가중치 # 경로에서 정적 장애물 회피하려면 최소 w_s : w_sm = 1 : 1 필요
        w_sm = 3.0  # 부드러움 비용 가중치 # 곡선 구간에서 인코스를 방지하려면 최소 w_sm : w_g = 3 : 1 필요
        w_g = 1.0   # 전역 경로 추종 비용 가중치
        w_d = 0.0   # 동적 장애물 비용

        # 1) 정적 장애물 비용 (각 후보 경로마다 값이 계산됨)
        set_c_s = self.compute_static_obstacle_cost(
            candidate_paths,
            len(candidate_paths),
            threshold=1.0,
            sigma=1.0
        )
        # 2) 부드러움 비용
        set_c_sm = []
        for candidate_path in candidate_paths:
            c_sm = self.compute_smoothness_cost_xy(candidate_path)
            set_c_sm.append(c_sm)

        # 3) 전역 경로 추종 비용
        set_c_g = self.compute_global_path_cost(candidate_paths)

        # 4) 동적 장애물 비용
        # set_c_d = self.compute_dynamic_cost(candidate_paths)
        # set_c_d = []
        # for candidate_path in candidate_paths:
        #     c_d = self.compute_dynamic_cost(candidate_path)
        #     set_c_d.append(c_d)

        # 4) 총 비용 계산
        c_totals = []
        for i in range(len(candidate_paths)):
            c_total = (w_s  * set_c_s[i]
                        + w_sm * set_c_sm[i]
                        + w_g  * set_c_g[i])
                        # + w_d  * set_c_d[i])  # 동적 장애물 비용 미구현이므로 0
            c_totals.append(c_total)
            rospy.loginfo("Candidate %d: c_static=%.3f, c_smooth=%.3f, c_global=%.3f, total_cost=%.3f", i, w_s  * set_c_s[i], w_sm * set_c_sm[i], w_g  * set_c_g[i], c_total)

        # 5) 최적 경로 선택 (가장 비용이 낮은 인덱스)
        min_idx = np.argmin(c_totals)
        best_path = candidate_paths[min_idx]
        best_cost = c_totals[min_idx]

        # Frenet -> Cartesian 변환
        best_cartesian_path = self.convert_frenet_path_to_cartesian(best_path)

        return best_cartesian_path,min_idx

    def compute_static_obstacle_cost(self, candidate_paths, number_of_candidate_paths, threshold=1.0, sigma=1.0):
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
            if len(self.obstacles_s) > 0:
                min_distances = [min(np.hypot(obs[0] - s, obs[1] - q) for obs in self.obstacles_s) for s, q in points]
                # 임계값보다 작은 값이 하나라도 있으면 1, 아니면 0 (max() 사용)
                indicators[pathnum] = max(1 if d < threshold else 0 for d in min_distances)
            else:
                indicators[pathnum] = 0  # 장애물이 없으면 0 유지

        # 3. 가우시안 필터를 적용하여 비용을 부드럽게 조정
        cost_profile = gaussian_filter1d(indicators, sigma=sigma)

        return cost_profile

    def compute_smoothness_cost_xy(self, frenet_path):
        """
        1) frenet_path (s,q) -> (X,Y)로 변환
        2) XY 좌표에서 곡률(kappa)을 구해, kappa^2 * ds 적분 근사
        3) 반환값이 작을수록 실제 XY 평면에서 '물리적'으로 부드러운 경로
        """

        import numpy as np

        # (A) Frenet -> Cartesian 변환
        xy_points = []
        for pose in frenet_path.poses:
            s_val = pose.pose.position.x
            q_val = pose.pose.position.y
            X, Y = self.frenet_to_cartesian(s_val, q_val)  # 전역 좌표계로 변환
            xy_points.append((X, Y))

        xy_points = np.array(xy_points)  # shape (N, 2)
        N = len(xy_points)
        if N < 3:
            return 0.0

        # (B) 곡률 계산에 필요한 구간 길이 ds, 곡률 근사
        total_cost = 0.0
        for i in range(1, N-1):
            x_prev, y_prev = xy_points[i-1]
            x_curr, y_curr = xy_points[i]
            x_next, y_next = xy_points[i+1]

            # ds_prev, ds_next
            ds_prev = np.hypot(x_curr - x_prev, y_curr - y_prev)
            ds_next = np.hypot(x_next - x_curr, y_next - y_curr)
            ds_avg = 0.5*(ds_prev + ds_next)

            # 곡률 근사 (삼각형 외접원 방식)
            kappa = self.approx_curvature(x_prev, y_prev, x_curr, y_curr, x_next, y_next)
            total_cost += (kappa**2)*ds_avg

        return total_cost

    def approx_curvature(self, x_prev, y_prev, x_curr, y_curr, x_next, y_next):
        """
        세 점을 이용해 곡률(kappa)을 근사:
        kappa ~ 2*sin(theta) / chord
        theta = angle between vec(a) and vec(b)
        chord = dist between first & last point
        """
        a = np.array([x_curr - x_prev, y_curr - y_prev])
        b = np.array([x_next - x_curr, y_next - y_curr])
        dot_ab = a.dot(b)
        cross_ab = a[0]*b[1] - a[1]*b[0]
        mag_a = np.hypot(a[0], a[1])
        mag_b = np.hypot(b[0], b[1])
        if mag_a < 1e-9 or mag_b < 1e-9:
            return 0.0
        sin_theta = abs(cross_ab)/(mag_a*mag_b)
        chord = np.hypot(x_next - x_prev, y_next - y_prev)
        if chord < 1e-9:
            return 0.0
        kappa = 2.0*sin_theta/chord
        return kappa

    def compute_global_path_cost(self, candidate_paths):
        """
        논문 식 (9)에 따른 전역 경로 추종 비용 C_g[i]를 계산한다.
        여기서는 각 경로의 '전역 경로로부터의 횡방향 차이'를 하나의 대표값(평균 등)으로 구하고,
        모든 후보 경로의 값을 합산한 뒤, i-th 경로의 값을 그 합으로 나누어 정규화한다.

        candidate_paths: List[nav_msgs/Path],
        각 Path는 Frenet 좌표 (s,q)로 저장되었다고 가정 (pose.position.x=s, pose.position.y=q)

        반환값:
        cg_list: 길이 len(candidate_paths)의 리스트,
                각 원소는 [0,1] 범위의 비용 (값이 작을수록 전역 경로 추종성이 좋음)
        """
        offsets = []
        for path in candidate_paths:
            # path 내 모든 점의 |q|를 합산(혹은 평균) -> 이 경로의 전역 경로 편차 지표
            # 간단히 합산 예시:
            sum_q = 0.0
            for pose in path.poses:
                q_val = pose.pose.position.y  # Frenet 좌표: y=q
                sum_q += abs(q_val)
            # 평균을 쓰고 싶다면 sum_q / len(path.poses)
            offsets.append(sum_q)

        total_offset = sum(offsets)
        if total_offset < 1e-9:
            # 모든 경로가 전역 경로에 거의 밀착된 경우 -> 비용을 동일하게 0으로
            # 혹은 모든 경로를 1/len(...)으로 할당할 수도 있음
            return [0.0]*len(candidate_paths)

        # 각 경로의 offset / 전체 offset -> [0,1] 정규화
        cg_list = [offset / total_offset for offset in offsets]
        return cg_list

    def compute_dynamic_cost(self, candidate_paths):
        """
        동적 장애물 비용을 계산하는 함수.

        각 후보 경로(candidate_path)에 대해:
        1. 후보 경로의 마지막 점의 s값을 s_c[i]로 보고,
            차량이 그 지점까지 주행하는 시간: t_veh[i] = s_c[i] / v_veh.
        2. self.dynamic_obstacles에 있는 각 동적 장애물(각각 (s, q, v_s))에 대해,
            장애물이 후보 경로 충돌 지점까지 도달하는 시간 t_obs_candidate = (s_c[i] - s_obs) / v_obs (s_obs < s_c인 경우) 를 계산.
            t_obs[i]는 이들 중 최소값.
        3. Δt[i] = t_obs[i] - t_veh[i]를 계산하여,
            - Δt[i] > 0 인 경우: 끼어들기(cut-in) 상황
            - Δt[i] <= 0 인 경우: 따라가기(follow) 상황 (Δt[i] = 0인 경우도 follow로 처리)
        4. [Cut-in 시나리오 (식 12, 13)]
            - delta_cut = s_c[i] + L_cut_in - v_veh * t_obs[i]
            - if delta_cut <= 0: a_req = 0
            else: a_req = 2 * delta_cut / (t_obs[i])^2
            - 동적 비용: C_d[i] = |a_req| * (s_c[i] + L_cut_in)
        5. [Follow 시나리오 (식 14, 15, 16)]
            - L_follow = L_0 if L_0 <= s_c[i] else s_c[i]
            - delta_follow = s_c[i] - L_follow - v_veh * t_obs[i]
            - if delta_follow <= 0: a_req = 0
            else: a_req = 2 * delta_follow / (t_obs[i])^2
            - 동적 비용: C_d[i] = |a_req| * (s_c[i] - L_follow)

        Returns:
        dynamic_costs: 각 후보 경로별 동적 장애물 비용의 리스트.
        """
        dynamic_costs = []
        L_cut_in = 2.0  # 끼어들기 위해 필요한 추가 거리 (예시)
        L0 = 2.0       # 기본 추종 거리 (예시)

        # 현재 차량 속도 (m/s); 너무 작으면 1e-3로 처리
        v_veh = self.current_speed if self.current_speed > 1e-3 else 1e-3

        for candidate_path in candidate_paths:
            # 1. 후보 경로의 로컬 길이 s_c 계산 (각 (s,q) 점 간 유클리드 거리 누적)
            s_c = 0.0
            poses = candidate_path.poses
            for i in range(1, len(poses)):
                s_prev = poses[i-1].pose.position.x
                q_prev = poses[i-1].pose.position.y
                s_curr = poses[i].pose.position.x
                q_curr = poses[i].pose.position.y
                s_c += np.hypot(s_curr - s_prev, q_curr - q_prev)
            t_veh = s_c / v_veh

            # 2. 동적 장애물 중, 후보 경로의 충돌(또는 추종) 지점에 해당하는 장애물 고려
            t_obs_candidates = []
            for obs in self.obstacles_s:
                # obs는 (s, q, v_s, v_q, a_s, a_q) 튜플로 가정
                obs_s = obs[0]
                v_obs = obs[2] if obs[2] > 1e-3 else 1e-3
                # 장애물이 후보 경로 충돌 지점 앞쪽에 있다면 (obs_s < s_c)
                if obs_s < s_c:
                    t_obs_candidate = (s_c - obs_s) / v_obs
                    t_obs_candidates.append(t_obs_candidate)
            if len(t_obs_candidates) == 0:
                t_obs_min = float('inf')
            else:
                t_obs_min = min(t_obs_candidates)

            # 3. 시간 차이 Δt
            delta_t = t_obs_min - t_veh

            if delta_t > 0:
                # Cut-in 시나리오: 차량이 도착하기 전에 장애물이 해당 지점에 도달함.
                delta_cut = s_c + L_cut_in - v_veh * t_obs_min
                if delta_cut <= 0:
                    a_req = 0.0
                else:
                    a_req = 2.0 * delta_cut / (t_obs_min**2)
                cost = abs(a_req) * (s_c + L_cut_in)
            else:
                # Follow 시나리오: Δt <= 0이면 차량이 먼저 도착하거나 딱 맞으므로, 장애물을 따라가야 함.
                L_follow = L0 if L0 <= s_c else s_c
                delta_follow = s_c - L_follow - v_veh * t_obs_min
                if delta_follow <= 0:
                    a_req = 0.0
                else:
                    a_req = 2.0 * delta_follow / (t_obs_min**2)
                cost = abs(a_req) * (s_c - L_follow)

            dynamic_costs.append(cost)
            # rospy.loginfo("Candidate: s_c=%.3f, t_veh=%.3f, t_obs=%.3f, Δt=%.3f, cost=%.3f", s_c, t_veh, t_obs_min, delta_t, cost)

        return dynamic_costs

if __name__ == '__main__':
    try:
        GlobalPathAnalyzer()
    except rospy.ROSInterruptException:
        pass

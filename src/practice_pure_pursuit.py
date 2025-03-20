#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, os
import numpy as np
from math import cos,sin,pi,sqrt,pow,atan2

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry,Path
from morai_msgs.msg import CtrlCmd
from tf.transformations import euler_from_quaternion


class pure_pursuit :
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        rospy.Subscriber("/local_path", Path, self.path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd_0',CtrlCmd, queue_size=1)

        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 2

        self.is_path = False
        self.is_odom = False

        self.forward_point = Point()
        self.current_postion = Point()
        self.is_look_forward_point = False
        self.vehicle_length = 1.63
        self.lfd = 3

        if self.vehicle_length is None or self.lfd is None:
            print("you need to change values at line 30~31 ,  self.vegicle_length , lfd")
            exit()
        rate = rospy.Rate(20) # 15hz
        while not rospy.is_shutdown():
            if self.is_path and self.is_odom:

                vehicle_position=self.current_postion
                self.is_look_forward_point= False

                translation=[vehicle_position.x, vehicle_position.y]

                t=np.array([  # 전역 좌표계를 기준으로 로봇의 방향과 평행이동을 표현(즉 로봇좌표계)
                        [cos(self.vehicle_yaw), -sin(self.vehicle_yaw),translation[0]],
                        [sin(self.vehicle_yaw),cos(self.vehicle_yaw),translation[1]],
                        [0                    ,0                    ,1            ]])

                det_t=np.array([ # 전역 좌표계를 기준으로 되어 있는 것을 로봇의 로컬 좌표계로 변환하기 위한 역변환 행렬
                       [t[0][0],t[1][0],-(t[0][0]*translation[0]+t[1][0]*translation[1])],
                       [t[0][1],t[1][1],-(t[0][1]*translation[0]+t[1][1]*translation[1])],
                       [0      ,0      ,1                                               ]])

                for i,value in enumerate(self.path.poses) : # 지역 경로
                    path_point=value.pose.position
                    global_path_point=[path_point.x,path_point.y,1]
                    local_path_point=det_t.dot(global_path_point)
                    if local_path_point[0]>0 : # 이 점이 차량의 전방에 위치한다면
                        dis=sqrt(pow(local_path_point[0],2)+pow(local_path_point[1],2))
                        if dis>= self.lfd:
                            self.forward_point=path_point
                            self.is_look_forward_point=True
                            break

                theta=atan2(local_path_point[1],local_path_point[0])

                if self.is_look_forward_point :
                    self.ctrl_cmd_msg.steering = atan2(2*self.vehicle_length*sin(theta),self.lfd)
                    if self.ctrl_cmd_msg.steering is None:
                        print("you need to change the value at line 70")
                        exit()
                    self.ctrl_cmd_msg.velocity=10.0
                    self.ctrl_cmd_msg.accel=0.50

                    os.system('clear')
                    print("-------------------------------------")
                    print(" steering (deg) = ", self.ctrl_cmd_msg.steering * 180/pi)
                    print(" velocity (kph) = ", self.ctrl_cmd_msg.velocity)
                    print("-------------------------------------")
                else :
                    print("no found forward point")
                    self.ctrl_cmd_msg.steering=0.0
                    self.ctrl_cmd_msg.velocity=0.0

                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

            else:
                os.system('clear')
                if not self.is_path:
                    print("[1] can't subscribe '/local_path' topic...")
                    pass
                if not self.is_odom:
                    print("[2] can't subscribe '/odom' topic...")
                    pass

            self.is_path = self.is_odom = False
            rate.sleep()

    def path_callback(self,msg):
        self.is_path=True
        self.path=msg
        print("path")

    def odom_callback(self,msg):
        print("odom")
        self.is_odom=True
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        _,_,self.vehicle_yaw=euler_from_quaternion(odom_quaternion)
        self.current_postion.x=msg.pose.pose.position.x
        self.current_postion.y=msg.pose.pose.position.y

if __name__ == '__main__':
    try:
        test_track=pure_pursuit()
    except rospy.ROSInterruptException:
        pass

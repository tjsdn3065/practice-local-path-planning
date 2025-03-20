#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
from math import sqrt,pow
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry,Path
import tf

class path_pub :

    def __init__(self):
        rospy.init_node('path_pub', anonymous=True)
        rospy.Subscriber("/odom", Odometry, self.odom_callback) # run gpsimu_parser.py
        rospy.Subscriber("/global_path",Path, self.global_Path_callback)

        self.local_path_pub = rospy.Publisher('/local_path',Path, queue_size=1)

        # 초기화
        self.global_path_msg=Path()
        self.global_path_msg.header.frame_id='/map'

        self.is_status=False

        self.local_path_size = 30

        self.x = 0
        self.y = 0


        rate = rospy.Rate(20) # 20hz
        while not rospy.is_shutdown():

            if self.is_status == True :
                local_path_msg=Path()
                local_path_msg.header.frame_id='/map'

                x=self.x
                y=self.y
                min_dis=float('inf')
                current_waypoint=-1
                for i,waypoint in enumerate(self.global_path_msg.poses) :
                    distance=sqrt(pow(x-waypoint.pose.position.x,2)+pow(y-waypoint.pose.position.y,2))
                    if distance < min_dis :
                        min_dis=distance
                        current_waypoint=i
                        self.prev_i=current_waypoint

                print(f'current_waypoint: {current_waypoint}, x: {self.global_path_msg.poses[current_waypoint].pose.position.x}, y: {self.global_path_msg.poses[current_waypoint].pose.position.y}')


                if current_waypoint != -1 :
                    if current_waypoint + self.local_path_size < len(self.global_path_msg.poses):
                        for num in range(current_waypoint,current_waypoint + self.local_path_size ) :
                            tmp_pose=PoseStamped()
                            tmp_pose.pose.position.x=self.global_path_msg.poses[num].pose.position.x
                            tmp_pose.pose.position.y=self.global_path_msg.poses[num].pose.position.y
                            tmp_pose.pose.orientation.w=1
                            local_path_msg.poses.append(tmp_pose)

                    else :
                        for num in range(current_waypoint,len(self.global_path_msg.poses) ) :
                            tmp_pose=PoseStamped()
                            tmp_pose.pose.position.x=self.global_path_msg.poses[num].pose.position.x
                            tmp_pose.pose.position.y=self.global_path_msg.poses[num].pose.position.y
                            tmp_pose.pose.orientation.w=1
                            local_path_msg.poses.append(tmp_pose)

                self.local_path_pub.publish(local_path_msg)

            rate.sleep()


    def odom_callback(self, msg):
        self.is_status=True

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y


    def global_Path_callback(self,msg):
        self.global_path_msg = msg

if __name__ == '__main__':
    try:
        test_track=path_pub()
    except rospy.ROSInterruptException:
        pass

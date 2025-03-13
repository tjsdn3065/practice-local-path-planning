#ifndef _LOCAL_PATH_PUB_H_
#define _LOCAL_PATH_PUB_H_

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <math.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <spline.h>  // tk::spline 헤더 (https://kluge.in-chemnitz.de/opensource/spline/ 참고)
#include <morai_msgs/EgoVehicleStatus.h>
#include <morai_msgs/CandidatePaths.h>
#include <morai_msgs/ObjectStatusList.h>

using namespace std;

class GlobalPathAnalyzer
{
public:
    GlobalPathAnalyzer(ros::NodeHandle &nh);
    void spin();

private:
    // NodeHandle
    ros::NodeHandle nh_;

    // Publishers
    ros::Publisher candidate_path_pub_;
    ros::Publisher candidate_path_pub1_;
    ros::Publisher candidate_path_pub2_;
    ros::Publisher candidate_path_pub3_;
    ros::Publisher candidate_path_pub4_;
    ros::Publisher candidate_path_pub5_;
    ros::Publisher candidate_path_pub6_;
    ros::Publisher candidate_path_pub7_;
    ros::Publisher optimal_path_pub_;

    // Subscribers
    ros::Subscriber inside_global_path_sub_;
    ros::Subscriber outside_global_path_sub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber obstacle_sub_;
    ros::Subscriber status_sub_;

    // Global path (inside)
    bool is_inside_global_path_ready_;
    vector<double> inside_s_vals_;
    double inside_total_length_;
    tk::spline inside_cs_x_;
    tk::spline inside_cs_y_;
    vector<double> inside_s_candidates_;

    // Global path (outside)
    bool is_outside_global_path_ready_;
    vector<double> outside_s_vals_;
    double outside_total_length_;
    tk::spline outside_cs_x_;
    tk::spline outside_cs_y_;
    vector<double> outside_s_candidates_;

    // 선택된 경로: 0 for inside, 1 for outside
    int choiced_path_;

    double sub_q_;

    // Odom and status
    bool is_odom_received_;
    double current_x_, current_y_, current_yaw_, current_speed_;
    bool is_status_;

    // 장애물 정보 (Frenet: (s,q) 저장)
    bool is_obstacle_;
    vector<pair<double, double>> obstacles_s_;

    // Callback functions
    void insideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg);
    void outsideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg);
    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void obstacleCallback(const morai_msgs::ObjectStatusList::ConstPtr &msg);
    void statusCallback(const morai_msgs::EgoVehicleStatus::ConstPtr &msg);

    // Utility functions for inside path
    double insideFindClosestSNewton(double x0, double y0);
    double insideDistSqGrad(double s, double x0, double y0);
    double insideDistSqHess(double s, double x0, double y0);
    double insideSignedLateralOffset(double x0, double y0, double s0);

    // Utility functions for outside path
    double outsideFindClosestSNewton(double x0, double y0);
    double outsideDistSqGrad(double s, double x0, double y0);
    double outsideDistSqHess(double s, double x0, double y0);
    double outsideSignedLateralOffset(double x0, double y0, double s0);

    // compute_s_coordinate (uses precomputed s_candidates)
    double computeSCoordinate(double x, double y); // 0 for inside, 1 for outside
    std::pair<double, double> compute_obstacle_frenet_all(double x_obs, double y_obs);

    // Frenet -> Cartesian conversion
    void frenetToCartesian(double s, double q, double &X, double &Y);

    void generateCandidatePaths(double s0, double q0, vector<nav_msgs::Path> &candidate_paths, vector<nav_msgs::Path> &cart_candidate_paths);
    void generateLocalPath(double s0, double q0, double lane_offset, nav_msgs::Path &path_msg, nav_msgs::Path &cart_path_msg);

    double compute_delta_s_vel(double v, double a_min, double s_min, double s_max);
    double compute_delta_s_with_obstacles(double s_vehicle, double s_vel, double s_min);
    double normalize_angle(double angle);
    void solve_cubic_spline_coeffs(double q_i, double dq_i, double q_f, double dq_f, double ds, double &a_, double &b_, double &c_, double &d_);
    double eval_q_spline_t(double a_, double b_, double c_, double d_, double t);
    nav_msgs::Path convert_frenet_path_to_cartesian(const nav_msgs::Path &frenet_path);

    std::pair<nav_msgs::Path, int> computeOptimalPath(const std::vector<nav_msgs::Path>& candidate_paths);
    std::vector<double> computeStaticObstacleCost(const std::vector<nav_msgs::Path>& candidate_paths, double threshold);
    std::vector<double> gaussian_filter1d(const std::vector<double>& input, double sigma);
    double computeSmoothnessCostXY(const nav_msgs::Path &frenet_path);
    double approxCurvature(double x_prev, double y_prev, double x_curr, double y_curr, double x_next, double y_next);
    std::vector<double> computeGlobalPathCost(const std::vector<nav_msgs::Path>& candidate_paths);
    std::vector<double> computeDynamicCost(const std::vector<nav_msgs::Path>& candidate_paths);
};

#endif

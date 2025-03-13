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
    std::vector<double> computeStaticObstacleCost(const std::vector<nav_msgs::Path>& candidate_paths, double threshold, double sigma);
    double computeSmoothnessCostXY(const nav_msgs::Path &frenet_path);
    double approxCurvature(double x_prev, double y_prev, double x_curr, double y_curr, double x_next, double y_next);
    std::vector<double> computeGlobalPathCost(const std::vector<nav_msgs::Path>& candidate_paths);
    std::vector<double> computeDynamicCost(const std::vector<nav_msgs::Path>& candidate_paths);
};

// -------------------- Constructor --------------------
GlobalPathAnalyzer::GlobalPathAnalyzer(ros::NodeHandle &nh) : nh_(nh)
{
    // Initialize flags
    is_inside_global_path_ready_ = false;
    is_outside_global_path_ready_ = false;
    is_odom_received_ = false;
    is_status_ = false;
    is_obstacle_ = false;
    choiced_path_ = -1;
    sub_q_ = 0;
    current_x_ = current_y_ = current_yaw_ = current_speed_ = 0.0;

    // Publishers
    candidate_path_pub_ = nh_.advertise<morai_msgs::CandidatePaths>("/candidate_paths", 1);
    candidate_path_pub1_ = nh_.advertise<nav_msgs::Path>("/candidate_path1", 1);
    candidate_path_pub2_ = nh_.advertise<nav_msgs::Path>("/candidate_path2", 1);
    candidate_path_pub3_ = nh_.advertise<nav_msgs::Path>("/candidate_path3", 1);
    candidate_path_pub4_ = nh_.advertise<nav_msgs::Path>("/candidate_path4", 1);
    candidate_path_pub5_ = nh_.advertise<nav_msgs::Path>("/candidate_path5", 1);
    candidate_path_pub6_ = nh_.advertise<nav_msgs::Path>("/candidate_path6", 1);
    candidate_path_pub7_ = nh_.advertise<nav_msgs::Path>("/candidate_path7", 1);
    optimal_path_pub_ = nh_.advertise<nav_msgs::Path>("/local_path", 1);

    // Subscribers
    inside_global_path_sub_ = nh_.subscribe("/inside_global_path", 1, &GlobalPathAnalyzer::insideGlobalPathCallback, this);
    outside_global_path_sub_ = nh_.subscribe("/outside_global_path", 1, &GlobalPathAnalyzer::outsideGlobalPathCallback, this);
    odom_sub_ = nh_.subscribe("/odom", 1, &GlobalPathAnalyzer::odomCallback, this);
    obstacle_sub_ = nh_.subscribe("/Object_topic", 1, &GlobalPathAnalyzer::obstacleCallback, this);
    status_sub_ = nh_.subscribe("Ego_topic", 1, &GlobalPathAnalyzer::statusCallback, this);
}

// -------------------- Main Loop --------------------
void GlobalPathAnalyzer::spin()
{
    ros::Rate rate(20);
    while (ros::ok())
    {
        ros::spinOnce();
        // os.system('clear') 대신 콘솔 출력 클리어 가능 (여기서는 생략)
        if (is_odom_received_ && is_inside_global_path_ready_ && is_outside_global_path_ready_ && is_status_ && is_obstacle_)
        {
            // inside_s_candidates 및 outside_s_candidates 초기화 (if empty)
            if (inside_s_candidates_.empty())
            {
                int num_samples = 3000;
                inside_s_candidates_.resize(num_samples);
                double step = inside_total_length_ / (num_samples - 1);
                for (int i = 0; i < num_samples; i++)
                    inside_s_candidates_[i] = i * step;
            }
            if (outside_s_candidates_.empty())
            {
                int num_samples = 3000;
                outside_s_candidates_.resize(num_samples);
                double step = outside_total_length_ / (num_samples - 1);
                for (int i = 0; i < num_samples; i++)
                    outside_s_candidates_[i] = i * step;
            }

            double inside_s0 = insideFindClosestSNewton(current_x_, current_y_);
            double outside_s0 = outsideFindClosestSNewton(current_x_, current_y_);
            double inside_q0 = insideSignedLateralOffset(current_x_, current_y_, inside_s0);
            double outside_q0 = outsideSignedLateralOffset(current_x_, current_y_, outside_s0);

            ROS_INFO("inside_q0: %.3f, outside_q0: %.3f", inside_q0, outside_q0);
            sub_q_ = fabs(inside_q0 - outside_q0);
            ROS_INFO("sub_q: %.3f", sub_q_);

            double s0, q0;
            if (fabs(inside_q0) <= fabs(outside_q0))
            {
                s0 = inside_s0;
                q0 = inside_q0;
                choiced_path_ = 0;
            }
            else
            {
                s0 = outside_s0;
                q0 = outside_q0;
                choiced_path_ = 1;
            }
            ROS_INFO("Choiced path: %d", choiced_path_);
            ROS_INFO("s0: %.3f, q0: %.3f", s0, q0);

            // Candidate paths generation (구현은 별도; 여기서는 placeholder)
            vector<nav_msgs::Path> candidate_paths_list;
            vector<nav_msgs::Path> cart_candidate_paths_list;
            generateCandidatePaths(s0, q0, candidate_paths_list, cart_candidate_paths_list);

            // Create CandidatePaths message
            morai_msgs::CandidatePaths candidate_paths_msg;
            candidate_paths_msg.paths = candidate_paths_list;

            morai_msgs::CandidatePaths cart_candidate_paths_msg;
            cart_candidate_paths_msg.paths = cart_candidate_paths_list;

            std::pair<nav_msgs::Path, int> result = computeOptimalPath(candidate_paths_msg.paths);
            nav_msgs::Path optimal_path = result.first;
            int min_idx = result.second;
            optimal_path_pub_.publish(optimal_path);

            // Publish candidate paths excluding optimal one
            vector<ros::Publisher> all_publishers = { candidate_path_pub1_, candidate_path_pub2_,
                                                        candidate_path_pub3_, candidate_path_pub4_,
                                                        candidate_path_pub5_, candidate_path_pub6_,
                                                        candidate_path_pub7_ };
            vector<nav_msgs::Path> candidate_paths = cart_candidate_paths_msg.paths;
            vector<nav_msgs::Path> filtered_paths;
            vector<ros::Publisher> filtered_publishers;
            for (size_t i = 0; i < candidate_paths.size(); i++)
            {
                if ((int)i != min_idx)
                {
                    filtered_paths.push_back(candidate_paths[i]);
                    filtered_publishers.push_back(all_publishers[i]);
                }
            }
            for (size_t i = 0; i < filtered_paths.size(); i++)
            {
                filtered_publishers[i].publish(filtered_paths[i]);
            }

            // Reset flags
            is_odom_received_ = false;
            is_status_ = false;
            is_obstacle_ = false;
        }
        rate.sleep();
    }
}

// -------------------- Callback Implementations --------------------
void GlobalPathAnalyzer::insideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg)
{
    if (msg->poses.empty())
    {
        ROS_WARN("Received empty inside_global_path");
        return;
    }
    if (!is_inside_global_path_ready_)
    {
        vector<double> x_points, y_points;
        for (const auto &pose : msg->poses)
        {
            x_points.push_back(pose.pose.position.x);
            y_points.push_back(pose.pose.position.y);
        }
        double tol = 1e-8;
        if (fabs(x_points.front() - x_points.back()) > tol || fabs(y_points.front() - y_points.back()) > tol)
        {
            ROS_WARN("inside_global_path first and last points differ; forcing correction.");
            x_points.back() = x_points.front();
            y_points.back() = y_points.front();
        }
        inside_s_vals_.clear();
        inside_s_vals_.push_back(0.0);
        for (size_t i = 1; i < x_points.size(); i++)
        {
            double dist = hypot(x_points[i] - x_points[i-1], y_points[i] - y_points[i-1]);
            inside_s_vals_.push_back(inside_s_vals_.back() + dist);
        }
        inside_total_length_ = inside_s_vals_.back();
        // Create splines using tk::spline
        inside_cs_x_.set_points(inside_s_vals_, x_points); // periodic boundary can be handled externally if needed
        inside_cs_y_.set_points(inside_s_vals_, y_points);
        is_inside_global_path_ready_ = true;
    }
}

void GlobalPathAnalyzer::outsideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg)
{
    if (msg->poses.empty())
    {
        ROS_WARN("Received empty outside_global_path");
        return;
    }
    if (!is_outside_global_path_ready_)
    {
        vector<double> x_points, y_points;
        for (const auto &pose : msg->poses)
        {
            x_points.push_back(pose.pose.position.x);
            y_points.push_back(pose.pose.position.y);
        }
        double tol = 1e-8;
        if (fabs(x_points.front() - x_points.back()) > tol || fabs(y_points.front() - y_points.back()) > tol)
        {
            ROS_WARN("outside_global_path first and last points differ; forcing correction.");
            x_points.back() = x_points.front();
            y_points.back() = y_points.front();
        }
        vector<double> s_vals;
        s_vals.push_back(0.0);
        for (size_t i = 1; i < x_points.size(); i++)
        {
            double dist = hypot(x_points[i] - x_points[i-1], y_points[i] - y_points[i-1]);
            s_vals.push_back(s_vals.back() + dist);
        }
        outside_s_vals_ = s_vals;
        outside_total_length_ = s_vals.back();
        outside_cs_x_.set_points(outside_s_vals_, x_points);
        outside_cs_y_.set_points(outside_s_vals_, y_points);
        is_outside_global_path_ready_ = true;
    }
}

void GlobalPathAnalyzer::odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    is_odom_received_ = true;
    current_x_ = msg->pose.pose.position.x;
    current_y_ = msg->pose.pose.position.y;
    tf::Quaternion q;
    tf::quaternionMsgToTF(msg->pose.pose.orientation, q);
    double roll, pitch, yaw;
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
    current_yaw_ = yaw;
}

void GlobalPathAnalyzer::obstacleCallback(const morai_msgs::ObjectStatusList::ConstPtr &msg)
{
    if (!is_inside_global_path_ready_ && !is_outside_global_path_ready_)
    {
        ROS_WARN("Global path not ready in obstacle callback");
        return;
    }
    is_obstacle_ = true;
    vector<pair<double,double>> dyn_obs;
    for (const auto &obstacle : msg->obstacle_list)
    {
        double x_obs = obstacle.position.x;
        double y_obs = obstacle.position.y;
        double s_obs, q_obs;
        tie(s_obs, q_obs) = compute_obstacle_frenet_all(x_obs, y_obs);
        if (fabs(q_obs) <= 1.5)
        {
            dyn_obs.push_back(make_pair(s_obs, q_obs));
        }
    }
    obstacles_s_ = dyn_obs;
}

void GlobalPathAnalyzer::statusCallback(const morai_msgs::EgoVehicleStatus::ConstPtr &msg)
{
    is_status_ = true;
    current_speed_ = msg->velocity.x * 3.75;
}

// -------------------- Utility Functions --------------------

// For inside path
double GlobalPathAnalyzer::insideFindClosestSNewton(double x0, double y0)
{
    if (!is_inside_global_path_ready_)
        return 0.0;
    double s_current = inside_s_candidates_[0];
    // Vectorized initial guess using candidate points:
    double min_dist = 1e12;
    for (size_t i = 0; i < inside_s_candidates_.size(); i++) {
        double s_val = inside_s_candidates_[i];
        double dx = x0 - inside_cs_x_(s_val);
        double dy = y0 - inside_cs_y_(s_val);
        double d = dx*dx + dy*dy;
        if (d < min_dist) {
            min_dist = d;
            s_current = s_val;
        }
    }
    int max_iter = 30;
    double tol = 1e-6;
    double h = 1e-4;
    for (int iter = 0; iter < max_iter; iter++)
    {
        double fprime = insideDistSqGrad(s_current, x0, y0);
        double fsecond = insideDistSqHess(s_current, x0, y0);
        if (fabs(fsecond) < 1e-12)
            break;
        double step = -fprime / fsecond;
        s_current += step;
        s_current = fmod(s_current, inside_total_length_);
        if (s_current < 0) s_current += inside_total_length_;
        if (fabs(step) < tol)
            break;
    }
    return s_current;
}

double GlobalPathAnalyzer::insideDistSqGrad(double s, double x0, double y0)
{
    double h = 1e-4;
    double dx = x0 - inside_cs_x_(s);
    double dy = y0 - inside_cs_y_(s);
    double dxds = (inside_cs_x_(s+h) - inside_cs_x_(s-h)) / (2*h);
    double dyds = (inside_cs_y_(s+h) - inside_cs_y_(s-h)) / (2*h);
    return -2.0 * (dx * dxds + dy * dyds);
}

double GlobalPathAnalyzer::insideDistSqHess(double s, double x0, double y0)
{
    double h = 1e-4;
    double dx = x0 - inside_cs_x_(s);
    double dy = y0 - inside_cs_y_(s);
    double dxds = (inside_cs_x_(s+h) - inside_cs_x_(s-h)) / (2*h);
    double dyds = (inside_cs_y_(s+h) - inside_cs_y_(s-h)) / (2*h);
    double d2xds2 = (inside_cs_x_(s+h) - 2*inside_cs_x_(s) + inside_cs_x_(s-h)) / (h*h);
    double d2yds2 = (inside_cs_y_(s+h) - 2*inside_cs_y_(s) + inside_cs_y_(s-h)) / (h*h);
    double val = (-dxds*dxds + dx * d2xds2) + (-dyds*dyds + dy * d2yds2);
    return -2.0 * val;
}

double GlobalPathAnalyzer::insideSignedLateralOffset(double x0, double y0, double s0)
{
    double h = 1e-4;
    double dx = inside_cs_x_(s0);
    double dy = inside_cs_y_(s0);
    double dxds = (inside_cs_x_(s0+h) - inside_cs_x_(s0-h)) / (2*h);
    double dyds = (inside_cs_y_(s0+h) - inside_cs_y_(s0-h)) / (2*h);
    double dx_veh = x0 - dx;
    double dy_veh = y0 - dy;
    double cross_val = dxds * dy_veh - dyds * dx_veh;
    double q0 = sqrt(dx_veh*dx_veh + dy_veh*dy_veh);
    return (cross_val > 0) ? q0 : -q0;
}

// For outside path, similar implementations:
double GlobalPathAnalyzer::outsideFindClosestSNewton(double x0, double y0)
{
    if (!is_outside_global_path_ready_)
        return 0.0;
    double s_current = outside_s_candidates_[0];
    double min_dist = 1e12;
    for (size_t i = 0; i < outside_s_candidates_.size(); i++) {
        double s_val = outside_s_candidates_[i];
        double dx = x0 - outside_cs_x_(s_val);
        double dy = y0 - outside_cs_y_(s_val);
        double d = dx*dx + dy*dy;
        if (d < min_dist) {
            min_dist = d;
            s_current = s_val;
        }
    }
    int max_iter = 30;
    double tol = 1e-6;
    double h = 1e-4;
    for (int iter = 0; iter < max_iter; iter++)
    {
        double fprime = outsideDistSqGrad(s_current, x0, y0);
        double fsecond = outsideDistSqHess(s_current, x0, y0);
        if (fabs(fsecond) < 1e-12)
            break;
        double step = -fprime / fsecond;
        s_current += step;
        s_current = fmod(s_current, outside_total_length_);
        if (s_current < 0) s_current += outside_total_length_;
        if (fabs(step) < tol)
            break;
    }
    return s_current;
}

double GlobalPathAnalyzer::outsideDistSqGrad(double s, double x0, double y0)
{
    double h = 1e-4;
    double dx = x0 - outside_cs_x_(s);
    double dy = y0 - outside_cs_y_(s);
    double dxds = (outside_cs_x_(s+h) - outside_cs_x_(s-h)) / (2*h);
    double dyds = (outside_cs_y_(s+h) - outside_cs_y_(s-h)) / (2*h);
    return -2.0 * (dx * dxds + dy * dyds);
}

double GlobalPathAnalyzer::outsideDistSqHess(double s, double x0, double y0)
{
    double h = 1e-4;
    double dx = x0 - outside_cs_x_(s);
    double dy = y0 - outside_cs_y_(s);
    double dxds = (outside_cs_x_(s+h) - outside_cs_x_(s-h)) / (2*h);
    double dyds = (outside_cs_y_(s+h) - outside_cs_y_(s-h)) / (2*h);
    double d2xds2 = (outside_cs_x_(s+h) - 2*outside_cs_x_(s) + outside_cs_x_(s-h)) / (h*h);
    double d2yds2 = (outside_cs_y_(s+h) - 2*outside_cs_y_(s) + outside_cs_y_(s-h)) / (h*h);
    double val = (-dxds*dxds + dx * d2xds2) + (-dyds*dyds + dy * d2yds2);
    return -2.0 * val;
}

double GlobalPathAnalyzer::outsideSignedLateralOffset(double x0, double y0, double s0)
{
    double h = 1e-4;
    double dx = outside_cs_x_(s0);
    double dy = outside_cs_y_(s0);
    double dxds = (outside_cs_x_(s0+h) - outside_cs_x_(s0-h)) / (2*h);
    double dyds = (outside_cs_y_(s0+h) - outside_cs_y_(s0-h)) / (2*h);
    double dx_veh = x0 - dx;
    double dy_veh = y0 - dy;
    double cross_val = dxds * dy_veh - dyds * dx_veh;
    double q0 = sqrt(dx_veh*dx_veh + dy_veh*dy_veh);
    return (cross_val > 0) ? q0 : -q0;
}

double GlobalPathAnalyzer::computeSCoordinate(double x, double y)
{
    // pathChoice: 0 -> inside, 1 -> outside
    vector<double> xs, ys;
    vector<double> s_candidates;
    if (choiced_path_ == 0) {
        xs.resize(inside_s_candidates_.size());
        ys.resize(inside_s_candidates_.size());
        s_candidates = inside_s_candidates_;
        for (size_t i = 0; i < inside_s_candidates_.size(); i++) {
            xs[i] = inside_cs_x_(inside_s_candidates_[i]);
            ys[i] = inside_cs_y_(inside_s_candidates_[i]);
        }
    } else {
        xs.resize(outside_s_candidates_.size());
        ys.resize(outside_s_candidates_.size());
        s_candidates = outside_s_candidates_;
        for (size_t i = 0; i < outside_s_candidates_.size(); i++) {
            xs[i] = outside_cs_x_(outside_s_candidates_[i]);
            ys[i] = outside_cs_y_(outside_s_candidates_[i]);
        }
    }
    double min_dist = 1e12;
    double s_best = 0.0;
    for (size_t i = 0; i < s_candidates.size(); i++) {
        double d = pow(x - xs[i], 2) + pow(y - ys[i], 2);
        if (d < min_dist) {
            min_dist = d;
            s_best = s_candidates[i];
        }
    }
    return s_best;
}

// Frenet -> Cartesian conversion
void GlobalPathAnalyzer::frenetToCartesian(double s, double q, double &X, double &Y)
{
    double dx, dy, dxds, dyds;
    // Compute tangent using central difference:
    double h = 1e-4;
    if (choiced_path_ == 0) {
        dx = inside_cs_x_(fmod(s, inside_total_length_));
        dy = inside_cs_y_(fmod(s, inside_total_length_));
        dxds = (inside_cs_x_(s+h) - inside_cs_x_(s-h)) / (2*h);
        dyds = (inside_cs_y_(s+h) - inside_cs_y_(s-h)) / (2*h);
    } else {
        dx = outside_cs_x_(fmod(s, outside_total_length_));
        dy = outside_cs_y_(fmod(s, outside_total_length_));
        dxds = (outside_cs_x_(s+h) - outside_cs_x_(s-h)) / (2*h);
        dyds = (outside_cs_y_(s+h) - outside_cs_y_(s-h)) / (2*h);
    }
    double normT = hypot(dxds, dyds);
    if (normT < 1e-9)
    {
        X = dx;
        Y = dy;
        return;
    }
    dxds /= normT;
    dyds /= normT;
    // 법선 벡터: (-dyds, dxds)
    double nx = -dyds;
    double ny = dxds;
    X = dx + q * nx;
    Y = dy + q * ny;
}

// -------------------- Obstacle Frenet Conversion --------------------
pair<double,double> GlobalPathAnalyzer::compute_obstacle_frenet_all(double x_obs, double y_obs)
{
    double s_obs = computeSCoordinate(x_obs, y_obs);
    double q_obs;
    if (choiced_path_ == 0)
        q_obs = insideSignedLateralOffset(x_obs, y_obs, s_obs);
    else
        q_obs = outsideSignedLateralOffset(x_obs, y_obs, s_obs);
    return make_pair(s_obs, q_obs);
}

double GlobalPathAnalyzer::compute_delta_s_vel(double v, double a_min, double s_min, double s_max)
{
    // a_min은 음수이므로 abs(a_min)
    double s_candidate = s_min + (v * v / fabs(a_min));
    if (s_candidate < s_max)
        return s_candidate;
    else
        return s_max;
}

double GlobalPathAnalyzer::compute_delta_s_with_obstacles(double s_vehicle, double s_vel, double s_min)
{
    vector<double> dist_candidates;
    for (const auto &obs : obstacles_s_) { // obstacles_s_: vector of pair<double, double> (s,q)
        double s_obs = obs.first;
        double dist = s_obs - s_vehicle;
        if (dist > 0 && dist < 10)
            dist_candidates.push_back(dist);
    }
    if (dist_candidates.empty())
        return s_vel;
    else {
        double s_obs = *min_element(dist_candidates.begin(), dist_candidates.end());
        return max(s_obs, s_min);
    }
}

double GlobalPathAnalyzer::normalize_angle(double angle)
{
    double two_pi = 2 * M_PI;
    double a = fmod(angle + M_PI, two_pi);
    if (a < 0)
        a += two_pi;
    return a - M_PI;
}

void GlobalPathAnalyzer::generateCandidatePaths(double s0, double q0,
    vector<nav_msgs::Path>& candidate_paths,
    vector<nav_msgs::Path>& cart_candidate_paths)
{
    candidate_paths.clear();
    cart_candidate_paths.clear();

    // sub_q: 두 전역 경로 간 q 차이 (멤버 변수)
    if (sub_q_ <= 0.3) {
        for (double lane_offset = -1.0; lane_offset <= 1.0 + 1e-3; lane_offset += 0.4) {
            nav_msgs::Path path_msg, cart_path_msg;
            generateLocalPath(s0, q0, lane_offset, path_msg, cart_path_msg);
            candidate_paths.push_back(path_msg);
            cart_candidate_paths.push_back(cart_path_msg);
        }
    }
    else {
        if (choiced_path_ == 0) {
            for (double lane_offset = -2.0; lane_offset <= 0.0 + 1e-3; lane_offset += 0.4) {
                nav_msgs::Path path_msg, cart_path_msg;
                generateLocalPath(s0, q0, lane_offset, path_msg, cart_path_msg);
                candidate_paths.push_back(path_msg);
                cart_candidate_paths.push_back(cart_path_msg);
            }
        }
        else {
            for (double lane_offset = 0.0; lane_offset <= 2.0 + 1e-3; lane_offset += 0.4) {
                nav_msgs::Path path_msg, cart_path_msg;
                generateLocalPath(s0, q0, lane_offset, path_msg, cart_path_msg);
                candidate_paths.push_back(path_msg);
                cart_candidate_paths.push_back(cart_path_msg);
            }
        }
    }
}

void GlobalPathAnalyzer::generateLocalPath(double s0, double q0, double lane_offset,
  nav_msgs::Path &path_msg, nav_msgs::Path &cart_path_msg)
{
    path_msg.header.frame_id = "map";
    cart_path_msg.header.frame_id = "map";

    double s_vel = compute_delta_s_vel(current_speed_, -3.0, 5.0, 10.0);
    double final_delta_s = compute_delta_s_with_obstacles(s0, s_vel, 5.0);

    double dxds,dyds;
    double h = 1e-4;
    double path_yaw;
    if (choiced_path_ == 0) {
        dxds = (inside_cs_x_(s0+h) - inside_cs_x_(s0-h)) / (2*h);
        dyds = (inside_cs_y_(s0+h) - inside_cs_y_(s0-h)) / (2*h);
        path_yaw = atan2(dyds, dxds);
    } else {
        dxds = (outside_cs_x_(s0+h) - outside_cs_x_(s0-h)) / (2*h);
        dyds = (outside_cs_y_(s0+h) - outside_cs_y_(s0-h)) / (2*h);
        path_yaw = atan2(dyds, dxds);
    }
    double dtheta = normalize_angle(current_yaw_ - path_yaw);
    double q_i = q0;
    double dq_i = tan(dtheta);
    double q_f = lane_offset;
    double dq_f = 0.0;

    // 3차 스플라인 계수 계산: solve_cubic_spline_coeffs()를 호출하여 a, b, c, d 결정
    double a_, b_, c_, d_;
    solve_cubic_spline_coeffs(q_i, dq_i, q_f, dq_f, final_delta_s, a_, b_, c_, d_);

    // t 구간 샘플링 (예: 10개의 샘플)
    int num_samples = 10;
    vector<double> t_samples(num_samples);
    double dt = final_delta_s / (num_samples - 1);
    for (int i = 0; i < num_samples; i++) {
        t_samples[i] = i * dt;
    }

    for (double t : t_samples) {
        double q_val = eval_q_spline_t(a_, b_, c_, d_, t);
        double s_val;
        if (choiced_path_ == 0)
            s_val = fmod(s0 + t, inside_total_length_);
        else
            s_val = fmod(s0 + t, outside_total_length_);
        // Cartesian 변환
        double X, Y;
        frenetToCartesian(s_val, q_val, X, Y);

        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.pose.position.x = s_val;
        pose.pose.position.y = q_val;
        pose.pose.orientation.w = 1.0;
        path_msg.poses.push_back(pose);

        geometry_msgs::PoseStamped cart_pose;
        cart_pose.header.frame_id = "map";
        cart_pose.pose.position.x = X;
        cart_pose.pose.position.y = Y;
        cart_pose.pose.orientation.w = 1.0;
        cart_path_msg.poses.push_back(cart_pose);
    }
}

void GlobalPathAnalyzer::solve_cubic_spline_coeffs(double q_i, double dq_i, double q_f, double dq_f, double ds,
  double &a_, double &b_, double &c_, double &d_)
{
    d_ = q_i;     // q(0) = q_i
    c_ = dq_i;    // q'(0) = dq_i
    double X_f = ds;
    // 2x2 선형 시스템:
    // a_*X_f^3 + b_*X_f^2 = q_f - (c_*X_f + d_)
    // 3a_*X_f^2 + 2b_*X_f = dq_f - c_
    double A11 = X_f * X_f * X_f;
    double A12 = X_f * X_f;
    double A21 = 3 * X_f * X_f;
    double A22 = 2 * X_f;
    double B1 = q_f - (c_ * X_f + d_);
    double B2 = dq_f - c_;
    double det = A11 * A22 - A12 * A21;
    a_ = (B1 * A22 - A12 * B2) / det;
    b_ = (A11 * B2 - B1 * A21) / det;
}

double GlobalPathAnalyzer::eval_q_spline_t(double a_, double b_, double c_, double d_, double t)
{
    return a_ * pow(t, 3) + b_ * pow(t, 2) + c_ * t + d_;
}

nav_msgs::Path GlobalPathAnalyzer::convert_frenet_path_to_cartesian(const nav_msgs::Path &frenet_path)
{
    nav_msgs::Path cartesian_path = frenet_path;
    cartesian_path.poses.clear();
    for (const auto &pose : frenet_path.poses) {
        double s_val = pose.pose.position.x;
        double q_val = pose.pose.position.y;
        double X, Y;
        frenetToCartesian(s_val, q_val, X, Y);
        geometry_msgs::PoseStamped new_pose = pose;
        new_pose.pose.position.x = X;
        new_pose.pose.position.y = Y;
        cartesian_path.poses.push_back(new_pose);
    }
    return cartesian_path;
}

// computeOptimalPath 함수
std::pair<nav_msgs::Path, int> GlobalPathAnalyzer::computeOptimalPath(const std::vector<nav_msgs::Path>& candidate_paths)
{
    double w_s = 1.0, w_sm = 1.0, w_g = 1.0, w_d = 0.0; // w_d 미구현 // 3:3:1:1

    // 1) 정적 장애물 비용
    std::vector<double> cost_static = computeStaticObstacleCost(candidate_paths, 1.0, 1.0);

    // 2) 부드러움 비용
    std::vector<double> cost_smooth;
    for (const auto &path : candidate_paths) {
        cost_smooth.push_back(computeSmoothnessCostXY(path));
    }

    // 3) 전역 경로 추종 비용
    std::vector<double> cost_global = computeGlobalPathCost(candidate_paths);

    // 4) 총 비용 계산
    std::vector<double> total_cost(candidate_paths.size(), 0.0);
    for (size_t i = 0; i < candidate_paths.size(); i++) {
        total_cost[i] = w_s * cost_static[i] + w_sm * cost_smooth[i] + w_g * cost_global[i] + w_d * 0.0;
        ROS_INFO("Candidate %zu: c_static=%.3f, c_smooth=%.3f, c_global=%.3f, total_cost=%.3f",
                 i, w_s * cost_static[i], w_sm * cost_smooth[i], w_g * cost_global[i], total_cost[i]);
    }
    int min_idx = std::min_element(total_cost.begin(), total_cost.end()) - total_cost.begin();
    nav_msgs::Path best_path = candidate_paths[min_idx];
    nav_msgs::Path best_cartesian_path = convert_frenet_path_to_cartesian(best_path);
    return std::make_pair(best_cartesian_path, min_idx);
}

// computeStaticObstacleCost 함수
std::vector<double> GlobalPathAnalyzer::computeStaticObstacleCost(const std::vector<nav_msgs::Path>& candidate_paths,
                                                                    double threshold, double sigma)
{
    std::vector<double> indicators(candidate_paths.size(), 0.0);
    for (size_t pathnum = 0; pathnum < candidate_paths.size(); pathnum++) {
        const auto &candidate_path = candidate_paths[pathnum];
        std::vector<std::pair<double, double>> points;
        for (const auto &pose : candidate_path.poses) {
            double s = pose.pose.position.x;
            double q = pose.pose.position.y;
            points.push_back(std::make_pair(s, q));
        }
        double indicator = 0.0;
        if (!obstacles_s_.empty()) {
            for (const auto &p : points) {
                double s_val = p.first, q_val = p.second;
                double min_dist = std::numeric_limits<double>::max();
                for (const auto &obs : obstacles_s_) {
                    double d = hypot(obs.first - s_val, obs.second - q_val);
                    if (d < min_dist)
                        min_dist = d;
                }
                if (min_dist < threshold) {
                    indicator = 1.0;
                    break;
                }
            }
        }
        indicators[pathnum] = indicator;
    }
    // Gaussian smoothing은 여기서는 구현하지 않음.
    return indicators;
}

// computeSmoothnessCostXY 함수
double GlobalPathAnalyzer::computeSmoothnessCostXY(const nav_msgs::Path &frenet_path)
{
    std::vector<std::pair<double, double>> xy_points;
    for (const auto &pose : frenet_path.poses) {
        double s_val = pose.pose.position.x;
        double q_val = pose.pose.position.y;
        double X, Y;
        // choiced_path_를 전달하여 내부/외부 경로에 따라 처리
        frenetToCartesian(s_val, q_val, X, Y);
        xy_points.push_back(std::make_pair(X, Y));
    }
    size_t N = xy_points.size();
    if (N < 3)
        return 0.0;
    double total_cost = 0.0;
    for (size_t i = 1; i < N - 1; i++) {
        double x_prev = xy_points[i - 1].first, y_prev = xy_points[i - 1].second;
        double x_curr = xy_points[i].first, y_curr = xy_points[i].second;
        double x_next = xy_points[i + 1].first, y_next = xy_points[i + 1].second;
        double ds_prev = hypot(x_curr - x_prev, y_curr - y_prev);
        double ds_next = hypot(x_next - x_curr, y_next - y_curr);
        double ds_avg = 0.5 * (ds_prev + ds_next);
        double kappa = approxCurvature(x_prev, y_prev, x_curr, y_curr, x_next, y_next);
        total_cost += (kappa * kappa) * ds_avg;
    }
    return total_cost;
}

// approxCurvature 함수
double GlobalPathAnalyzer::approxCurvature(double x_prev, double y_prev, double x_curr, double y_curr, double x_next, double y_next)
{
    double ax = x_curr - x_prev, ay = y_curr - y_prev;
    double bx = x_next - x_curr, by = y_next - y_curr;
    double dot_ab = ax * bx + ay * by;
    double cross_ab = ax * by - ay * bx;
    double mag_a = hypot(ax, ay);
    double mag_b = hypot(bx, by);
    if (mag_a < 1e-9 || mag_b < 1e-9)
        return 0.0;
    double sin_theta = fabs(cross_ab) / (mag_a * mag_b);
    double chord = hypot(x_next - x_prev, y_next - y_prev);
    if (chord < 1e-9)
        return 0.0;
    double kappa = 2.0 * sin_theta / chord;
    return kappa;
}

// computeGlobalPathCost 함수
std::vector<double> GlobalPathAnalyzer::computeGlobalPathCost(const std::vector<nav_msgs::Path>& candidate_paths)
{
    std::vector<double> offsets;
    for (const auto &path : candidate_paths) {
        double sum_q = 0.0;
        for (const auto &pose : path.poses) {
            double q_val = pose.pose.position.y;
            sum_q += fabs(q_val);
        }
        offsets.push_back(sum_q);
    }
    double total_offset = 0.0;
    for (double v : offsets)
        total_offset += v;
    std::vector<double> cg_list;
    if (total_offset < 1e-9) {
        cg_list.resize(candidate_paths.size(), 0.0);
        return cg_list;
    }
    for (double v : offsets) {
        cg_list.push_back(v / total_offset);
    }
    return cg_list;
}

// computeDynamicCost 함수
std::vector<double> GlobalPathAnalyzer::computeDynamicCost(const std::vector<nav_msgs::Path>& candidate_paths)
{
    std::vector<double> dynamic_costs;
    double L_cut_in = 2.0; // 끼어들기에 필요한 추가 거리
    double L0 = 2.0;      // 기본 추종 거리
    double v_veh = (current_speed_ > 1e-3) ? current_speed_ : 1e-3;

    for (const auto &candidate_path : candidate_paths) {
        // 1. 후보 경로의 로컬 길이 s_c 계산 (각 (s,q) 점 간 유클리드 거리 누적)
        double s_c = 0.0;
        const auto &poses = candidate_path.poses;
        for (size_t i = 1; i < poses.size(); i++) {
            double s_prev = poses[i-1].pose.position.x;
            double q_prev = poses[i-1].pose.position.y;
            double s_curr = poses[i].pose.position.x;
            double q_curr = poses[i].pose.position.y;
            s_c += hypot(s_curr - s_prev, q_curr - q_prev);
        }
        double t_veh = s_c / v_veh;

        // 2. 장애물들 중, 후보 경로의 충돌(또는 추종) 지점에 해당하는 장애물 고려.
        // obstacles_s_는 vector<pair<double,double>> (s,q)로 저장되어 있음.
        std::vector<double> t_obs_candidates;
        for (const auto &obs : obstacles_s_) {
            double obs_s = obs.first;
            // 동적 장애물 속도 정보가 없으므로 기본 v_obs 사용
            double v_obs = 1e-3;
            if (obs_s < s_c) {
                double t_obs_candidate = (s_c - obs_s) / v_obs;
                t_obs_candidates.push_back(t_obs_candidate);
            }
        }
        double t_obs_min = t_obs_candidates.empty() ? INFINITY : *std::min_element(t_obs_candidates.begin(), t_obs_candidates.end());
        double delta_t = t_obs_min - t_veh;
        double cost = 0.0, a_req = 0.0;
        if (delta_t > 0) {
            // Cut-in 시나리오
            double delta_cut = s_c + L_cut_in - v_veh * t_obs_min;
            if (delta_cut <= 0)
                a_req = 0.0;
            else
                a_req = 2.0 * delta_cut / (t_obs_min * t_obs_min);
            cost = fabs(a_req) * (s_c + L_cut_in);
        }
        else {
            // Follow 시나리오
            double L_follow = (L0 <= s_c) ? L0 : s_c;
            double delta_follow = s_c - L_follow - v_veh * t_obs_min;
            if (delta_follow <= 0)
                a_req = 0.0;
            else
                a_req = 2.0 * delta_follow / (t_obs_min * t_obs_min);
            cost = fabs(a_req) * (s_c - L_follow);
        }
        dynamic_costs.push_back(cost);
        // ROS_INFO("Candidate: s_c=%.3f, t_veh=%.3f, t_obs=%.3f, delta_t=%.3f, cost=%.3f",
        //          s_c, t_veh, t_obs_min, delta_t, cost);
    }
    return dynamic_costs;
}

// -------------------- Main --------------------
int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_path_pub");
    ros::NodeHandle nh;
    GlobalPathAnalyzer analyzer(nh);
    analyzer.spin();
    return 0;
}

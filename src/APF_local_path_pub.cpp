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
#include <morai_msgs/EgoVehicleStatus.h>
#include <morai_msgs/CandidatePaths.h>
#include <morai_msgs/ObjectStatusList.h>

using namespace std;

class APFLocalPathPlannning
{
public:
    APFLocalPathPlannning(ros::NodeHandle &nh);
    void spin();

private:
    // NodeHandle
    ros::NodeHandle nh_;

    // Publishers
    ros::Publisher optimal_path_pub_;

    // Subscribers
    ros::Subscriber inside_global_path_sub_;
    ros::Subscriber outside_global_path_sub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber obstacle_sub_;
    ros::Subscriber status_sub_;

    // 선택된 경로: 0 for inside, 1 for outside
    int choiced_path_;

    vector<pair<double, double>> obstacles_;

    // Odom and status
    bool is_odom_received_;
    double current_x_, current_y_, current_yaw_, current_speed_;
    bool is_status_;

    // Callback functions
    void insideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg);
    void outsideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg);
    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void obstacleCallback(const morai_msgs::ObjectStatusList::ConstPtr &msg);
    void statusCallback(const morai_msgs::EgoVehicleStatus::ConstPtr &msg);
};

// -------------------- Constructor --------------------
APFLocalPathPlannning::GlobalPathAnalyzer(ros::NodeHandle &nh) : nh_(nh)
{
    // Initialize flags
    is_inside_global_path_ready_ = false;
    is_outside_global_path_ready_ = false;
    is_odom_received_ = false;
    is_status_ = false;
    is_obstacle_ = false;

    current_x_ = current_y_ = current_yaw_ = current_speed_ = 0.0;

    // Publishers
    optimal_path_pub_ = nh_.advertise<nav_msgs::Path>("/local_path", 1);

    // Subscribers
    inside_global_path_sub_ = nh_.subscribe("/inside_global_path", 1, &APFLocalPathPlannning::insideGlobalPathCallback, this);
    outside_global_path_sub_ = nh_.subscribe("/outside_global_path", 1, &APFLocalPathPlannning::outsideGlobalPathCallback, this);
    odom_sub_ = nh_.subscribe("/odom", 1, &APFLocalPathPlannning::odomCallback, this);
    obstacle_sub_ = nh_.subscribe("/Object_topic", 1, &APFLocalPathPlannning::obstacleCallback, this);
    status_sub_ = nh_.subscribe("Ego_topic", 1, &APFLocalPathPlannning::statusCallback, this);
}

// -------------------- Main Loop --------------------
void APFLocalPathPlannning::spin()
{
    ros::Rate rate(20);
    while (ros::ok())
    {
        ros::spinOnce();
        // os.system('clear') 대신 콘솔 출력 클리어 가능 (여기서는 생략)
        if (is_odom_received_ && is_inside_global_path_ready_ && is_outside_global_path_ready_ && is_status_ && is_obstacle_)
        {
            optimal_path_pub_.publish(optimal_path);
            // Reset flags
            is_odom_received_ = false;
            is_status_ = false;
            is_obstacle_ = false;
        }
        rate.sleep();
    }
}

// -------------------- Callback Implementations --------------------
void APFLocalPathPlannning::insideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg)
{
    if (msg->poses.empty())
    {
        ROS_WARN("Received empty inside_global_path");
        return;
    }
    vector<double> x_points, y_points;
    for (const auto &pose : msg->poses)
    {
        x_points.push_back(pose.pose.position.x);
        y_points.push_back(pose.pose.position.y);
    }
    double tol = 1e-8;
    if (fabs(x_points.front() - x_points.back()) > tol || fabs(y_points.front() - y_points.back()) > tol)
    {
        // 폐구간 보정
        ROS_WARN("inside_global_path first and last points differ; forcing correction.");
        x_points.back() = x_points.front();
        y_points.back() = y_points.front();
    }
}


void APFLocalPathPlannning::outsideGlobalPathCallback(const nav_msgs::Path::ConstPtr &msg)
{
    if (msg->poses.empty())
    {
        ROS_WARN("Received empty outside_global_path");
        return;
    }
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
}

void APFLocalPathPlannning::odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
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

void APFLocalPathPlannning::obstacleCallback(const morai_msgs::ObjectStatusList::ConstPtr &msg)
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
        dyn_obs.push_back(make_pair(x_obs, y_obs));
    }
    obstacles_ = dyn_obs;
}

void APFLocalPathPlannning::statusCallback(const morai_msgs::EgoVehicleStatus::ConstPtr &msg)
{
    is_status_ = true;
    current_speed_ = msg->velocity.x * 3.75;
}

// -------------------- Main --------------------
int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_path_pub");
    ros::NodeHandle nh;
    APFLocalPathPlannning local_path(nh);
    local_path.spin();
    return 0;
}

#include <local_path_pub.h>

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

    num_samples_ = 3000;    // 상황에 따라 수정

    inside_s0_ = 0.0;
    inside_q0_ = 0.0;

    outside_s0_ = 0.0;
    outside_q0_ = 0.0;

    s0_ = q0_ = 0.0;

    sub_q_ = 0;

    a_min_ = -3.0;    // 상황에 따라 수정
    s_min_ = 10.0;    // 상황에 따라 수정
    s_max_ = 20.0;    // 상황에 따라 수정

    future_inside_s0_ = 0;
    future_outside_s0_ = 0;
    future_in_X_ = future_in_Y_ = future_out_X_ = future_out_Y_ = 0;

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
                double step;
                inside_s_candidates_.resize(num_samples_);
                step = inside_total_length_ / (num_samples_ - 1);
                for (int i = 0; i < num_samples_; i++)
                    inside_s_candidates_[i] = i * step;
            }
            if (outside_s_candidates_.empty())
            {
                double step;
                outside_s_candidates_.resize(num_samples_);
                step = outside_total_length_ / (num_samples_ - 1);
                for (int i = 0; i < num_samples_; i++)
                    outside_s_candidates_[i] = i * step;
            }

            inside_s0_ = insideFindClosestSNewton(current_x_, current_y_);
            outside_s0_ = outsideFindClosestSNewton(current_x_, current_y_);
            inside_q0_ = insideSignedLateralOffset(current_x_, current_y_, inside_s0_);
            outside_q0_ = outsideSignedLateralOffset(current_x_, current_y_, outside_s0_);

            future_inside_s0_ = fmod(inside_s0_ + 20.0, inside_total_length_);
            future_outside_s0_ = fmod(outside_s0_ + 20.0, outside_total_length_);
            // Cartesian 변환
            frenetToCartesian(future_inside_s0_, 0, future_in_X_, future_in_Y_);
            frenetToCartesian(future_outside_s0_, 0, future_out_X_, future_out_Y_);

            ROS_INFO("inside_q0: %.3f, outside_q0: %.3f", inside_q0_, outside_q0_);
            sub_q_ = fabs(inside_q0_ - outside_q0_);
            ROS_INFO("sub_q: %.3f", sub_q_);

            if (fabs(inside_q0_) <= fabs(outside_q0_))
            {
                s0_ = inside_s0_;
                q0_ = inside_q0_;
                choiced_path_ = 0;
                future_inside_s0_ = fmod(inside_s0_ + 20.0, inside_total_length_);
            }
            else
            {
                s0_ = outside_s0_;
                q0_ = outside_q0_;
                choiced_path_ = 1;
                future_outside_s0_ = fmod(outside_s0_ + 20.0, outside_total_length_);
            }
            ROS_INFO("Choiced path: %d", choiced_path_);
            ROS_INFO("s0: %.3f, q0: %.3f", s0_, q0_);

            // Candidate paths generation (구현은 별도; 여기서는 placeholder)
            vector<nav_msgs::Path> candidate_paths_list;
            vector<nav_msgs::Path> cart_candidate_paths_list;
            generateCandidatePaths(s0_, q0_, candidate_paths_list, cart_candidate_paths_list);

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
            // 폐구간 보정
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
        inside_cs_x_.set_points(inside_s_vals_, x_points);
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
        double d = dx*dx + dy*dy;    // 실제 거리를 최소화 하는 것과, 제곱된 거리를 최소화 하는 것은 같은 최솟값의 위치를 준다.
        if (d < min_dist) {
            min_dist = d;
            s_current = s_val;
        }
    }
    int max_iter = 30;
    double tol = 1e-6;
    for (int iter = 0; iter < max_iter; iter++)
    {
        // f(x)는 거리 제곱 함수
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
    double dx = x0 - inside_cs_x_(s);
    double dy = y0 - inside_cs_y_(s);
    double dxds = inside_cs_x_.deriv(1, s);
    double dyds = inside_cs_y_.deriv(1, s);
    return -2.0 * (dx * dxds + dy * dyds);
}

double GlobalPathAnalyzer::insideDistSqHess(double s, double x0, double y0)
{
    double dx = x0 - inside_cs_x_(s);
    double dy = y0 - inside_cs_y_(s);
    double dxds = inside_cs_x_.deriv(1, s);
    double dyds = inside_cs_y_.deriv(1, s);
    double d2xds2 = inside_cs_x_.deriv(2, s);
    double d2yds2 = inside_cs_y_.deriv(2, s);
    double val = (-dxds*dxds + dx * d2xds2) + (-dyds*dyds + dy * d2yds2);
    return -2.0 * val;
}

double GlobalPathAnalyzer::insideSignedLateralOffset(double x0, double y0, double s0)
{
    double x_s0 = inside_cs_x_(s0);
    double y_s0 = inside_cs_y_(s0);
    double dxds = inside_cs_x_.deriv(1, s0);
    double dyds = inside_cs_y_.deriv(1, s0);
    double dx_veh = x0 - x_s0;
    double dy_veh = y0 - y_s0;
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
    double dx = x0 - outside_cs_x_(s);
    double dy = y0 - outside_cs_y_(s);
    double dxds = outside_cs_x_.deriv(1, s);
    double dyds = outside_cs_y_.deriv(1, s);
    return -2.0 * (dx * dxds + dy * dyds);
}

double GlobalPathAnalyzer::outsideDistSqHess(double s, double x0, double y0)
{
    double dx = x0 - outside_cs_x_(s);
    double dy = y0 - outside_cs_y_(s);
    double dxds = outside_cs_x_.deriv(1, s);
    double dyds = outside_cs_y_.deriv(1, s);
    double d2xds2 = outside_cs_x_.deriv(2, s);
    double d2yds2 = outside_cs_y_.deriv(2, s);
    double val = (-dxds*dxds + dx * d2xds2) + (-dyds*dyds + dy * d2yds2);
    return -2.0 * val;
}

double GlobalPathAnalyzer::outsideSignedLateralOffset(double x0, double y0, double s0)
{
    double x_s0 = outside_cs_x_(s0);
    double y_s0 = outside_cs_y_(s0);
    double dxds = outside_cs_x_.deriv(1, s0);
    double dyds = outside_cs_y_.deriv(1, s0);
    double dx_veh = x0 - x_s0;
    double dy_veh = y0 - y_s0;
    double cross_val = dxds * dy_veh - dyds * dx_veh;
    double q0 = sqrt(dx_veh*dx_veh + dy_veh*dy_veh);
    return (cross_val > 0) ? q0 : -q0;
}

// Frenet -> Cartesian conversion
void GlobalPathAnalyzer::frenetToCartesian(double s, double q, double &X, double &Y)
{
    double x_s, y_s, dxds, dyds;
    // Compute tangent using central difference:
    if (choiced_path_ == 0) {
        x_s = inside_cs_x_(s);
        y_s = inside_cs_y_(s);
        dxds = inside_cs_x_.deriv(1, s);
        dyds = inside_cs_y_.deriv(1, s);
    } else {
        x_s = outside_cs_x_(s);
        y_s = outside_cs_y_(s);
        dxds = outside_cs_x_.deriv(1, s);
        dyds = outside_cs_y_.deriv(1, s);
    }
    double normT = hypot(dxds, dyds);
    if (normT < 1e-9)
    {
        X = x_s;
        Y = y_s;
        return;
    }
    // 접선 벡터를 정규화하여 단위벡터로 만듬
    dxds /= normT;
    dyds /= normT;
    // 법선 벡터: (-dyds, dxds)
    double nx = -dyds;
    double ny = dxds;
    X = x_s + q * nx;
    Y = y_s + q * ny;
}

pair<double,double> GlobalPathAnalyzer::compute_obstacle_frenet_all(double obs_x0, double obs_y0)
{
    double obs_s0;
    double obs_q0;
    if (choiced_path_ == 0)
    {
        obs_s0 = insideFindClosestSNewton(obs_x0, obs_y0);
        obs_q0 = insideSignedLateralOffset(obs_x0, obs_y0, obs_s0);
    }
    else
    {
        obs_s0 = outsideFindClosestSNewton(obs_x0, obs_y0);
        obs_q0 = outsideSignedLateralOffset(obs_x0, obs_y0, obs_s0);
    }
    return make_pair(obs_s0, obs_q0);
}

double GlobalPathAnalyzer::compute_delta_s_vel()
{
    // a_min은 음수이므로 abs(a_min)
    double s_candidate = s_min_ + (current_speed_ * current_speed_ / fabs(a_min_));
    if (s_candidate < s_max_)
        return s_candidate;
    else
        return s_max_;
}

double GlobalPathAnalyzer::compute_delta_s_with_obstacles(double s0, double delta_s)
{
    vector<double> dist_candidates;
    for (const auto &obs : obstacles_s_) { // obstacles_s_: vector of pair<double, double> (s,q)
        double obs_s0 = obs.first;
        double dist = obs_s0 - s0;
        if (dist > 0 && dist < 20)
            dist_candidates.push_back(dist);
    }
    if (dist_candidates.empty())
        return delta_s;
    else {
        double obs_s0 = *min_element(dist_candidates.begin(), dist_candidates.end());
        return max(obs_s0, s_min_);
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

    double future_inside_q0 = insideSignedLateralOffset(future_in_X_, future_in_Y_, future_inside_s0_);
    double future_outside_q0 = outsideSignedLateralOffset(future_out_X_, future_out_Y_, future_outside_s0_);

    ROS_INFO("future_inside_q0: %.3f", future_inside_q0);
    ROS_INFO("future_outside_q0: %.3f", future_outside_q0);

    // sub_q: 두 전역 경로 간 q 차이 (멤버 변수)
    if (sub_q_ <= 0.3) {
        for (double lane_offset = -1.5; lane_offset <= 1.5 + 1e-3; lane_offset += 0.5) {
            nav_msgs::Path path_msg, cart_path_msg;
            generateLocalPath(s0, q0, lane_offset, path_msg, cart_path_msg);
            candidate_paths.push_back(path_msg);
            cart_candidate_paths.push_back(cart_path_msg);
        }
    }
    else {
        if (choiced_path_ == 0) {
            for (double lane_offset = -2.0; lane_offset <= 0.4 + 1e-3; lane_offset += 0.4) {
                nav_msgs::Path path_msg, cart_path_msg;
                generateLocalPath(s0, q0, lane_offset, path_msg, cart_path_msg);
                candidate_paths.push_back(path_msg);
                cart_candidate_paths.push_back(cart_path_msg);
            }
        }
        else {
            for (double lane_offset = -0.4; lane_offset <= 2.0 + 1e-3; lane_offset += 0.4) {
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

    double delta_s = compute_delta_s_vel();
    double final_delta_s = compute_delta_s_with_obstacles(s0, delta_s);

    double dxds,dyds;
    double path_yaw;
    if (choiced_path_ == 0) {
        dxds = inside_cs_x_.deriv(1, s0);
        dyds = inside_cs_y_.deriv(1, s0);
        path_yaw = atan2(dyds, dxds);
    } else {
        dxds = outside_cs_x_.deriv(1, s0);
        dyds = outside_cs_y_.deriv(1, s0);
        path_yaw = atan2(dyds, dxds);
    }
    double dtheta = normalize_angle(current_yaw_ - path_yaw);
    double q_i = q0;
    double dq_i = tan(dtheta);
    double q_f = lane_offset;
    double dq_f = 0.0;

    // 3차 스플라인 계수 계산: solve_cubic_spline_coeffs()를 호출하여 a, b, c, d 결정
    double a, b, c, d;
    solve_cubic_spline_coeffs(q_i, dq_i, q_f, dq_f, final_delta_s, a, b, c, d);

    // t 구간 샘플링 (예: 10개의 샘플)
    int num_samples = 20;
    vector<double> t_samples(num_samples);
    double dt = final_delta_s / (num_samples - 1);
    for (int i = 0; i < num_samples; i++) {
        t_samples[i] = i * dt;
    }

    for (double t : t_samples) {
        double q_val = eval_q_spline_t(a, b, c, d, t);
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
  double &a, double &b, double &c, double &d)
{
    d = q_i;     // q(0) = q_i
    c = dq_i;    // q'(0) = dq_i
    double X_f = ds;
    // 2x2 선형 시스템:
    // a_*X_f^3 + b_*X_f^2 = q_f - (c_*X_f + d_)
    // 3a_*X_f^2 + 2b_*X_f = dq_f - c_
    double A11 = X_f * X_f * X_f;
    double A12 = X_f * X_f;
    double A21 = 3 * X_f * X_f;
    double A22 = 2 * X_f;
    double B1 = q_f - (c * X_f + d);
    double B2 = dq_f - c;
    double det = A11 * A22 - A12 * A21;
    a = (B1 * A22 - A12 * B2) / det;
    b = (A11 * B2 - B1 * A21) / det;
}

double GlobalPathAnalyzer::eval_q_spline_t(double a, double b, double c, double d, double t)
{
    return a * pow(t, 3) + b * pow(t, 2) + c * t + d;
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
    std::vector<double> cost_static = computeStaticObstacleCost(candidate_paths, 1.0);

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
    double threshold)
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
    // Gaussian smoothing 적용
    std::vector<double> cost_profile = gaussian_filter1d(indicators, 1.0);
    return cost_profile;
}

// 1차원 가우시안 필터 함수
std::vector<double> GlobalPathAnalyzer::gaussian_filter1d(const std::vector<double>& input, double sigma)
{
    int n = input.size();
    std::vector<double> output(n, 0.0);
    // 커널 반경: 보통 3*sigma 정도를 사용 (이보다 더 큰 값은 거의 0에 수렴)
    int radius = std::ceil(3 * sigma);
    int kernel_size = 2 * radius + 1;
    std::vector<double> kernel(kernel_size, 0.0);
    double sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        double val = std::exp(- (i * i) / (2 * sigma * sigma));
        kernel[i + radius] = val;
        sum += val;
    }
    // 커널 정규화
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    // 컨볼루션 수행 (경계는 클램핑)
    for (int i = 0; i < n; i++) {
        double conv = 0.0;
        for (int j = -radius; j <= radius; j++) {
            int idx = i + j;
            // 경계 클램핑: idx가 음수이면 0, n 이상이면 n-1로 설정
            idx = std::max(0, std::min(idx, n - 1));
            conv += input[idx] * kernel[j + radius];
        }
        output[i] = conv;
    }
    return output;
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

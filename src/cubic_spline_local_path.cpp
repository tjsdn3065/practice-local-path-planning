#include <cubic_spline_local_path.h>

// -------------------- Constructor --------------------
GlobalPathAnalyzer::GlobalPathAnalyzer(ros::NodeHandle &nh) : nh_(nh)
{
    // Initialize flags
    is_global_path_ready_ = false;
    lane_offsets_ready_ = false;
    publish_candidate_paths_ready_ = false;

    num_samples_ = 3000;    // 상황에 따라 수정

    total_length_ =  0.0;

    s0_ = q0_ = 0.0;

    a_min_ = -3.0;    // 상황에 따라 수정
    a_max_ = 3.0;    // 상황에 따라 수정
    s_min_ = 15.0;    // 상황에 따라 수정
    s_max_ = 20.0;    // 상황에 따라 수정

    current_x_ = current_y_ = current_yaw_ = current_speed_ = 0.0;

    // Publishers
    optimal_path_pub_ = nh_.advertise<nav_msgs::Path>("/local_path", 1);
    target_v_pub_      = nh_.advertise<std_msgs::Float32>("/target_v", 1);
    target_a_pub_      = nh_.advertise<std_msgs::Float32>("/target_a", 1);

    // Subscribers
    global_path_sub_ = nh_.subscribe("/global_path", 1, &GlobalPathAnalyzer::GlobalPathCallback, this);
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
        if (is_global_path_ready_)
        {
            // 시작 시각
            ros::WallTime t0 = ros::WallTime::now();

            s0_ = FindClosestSNewton(current_x_, current_y_);
            q0_ = SignedLateralOffset(current_x_, current_y_, s0_);

            ROS_INFO("s0: %.3f, q0: %.3f", s0_, q0_);

            // 경로 후보군 생성
            generateCandidatePaths(s0_, q0_, candidate_paths_list, cart_candidate_paths_list);

            // 경로 후보군 생성 소요 시간
            double dt_gen = (ros::WallTime::now() - t0).toSec();

            // 최적 경로 계산
            nav_msgs::Path optimal_path;
            int optimal_idx;
            double target_v, target_a;
            tie(optimal_path, optimal_idx, target_v, target_a) = computeOptimalPath(candidate_paths_list);

            // 최적 경로 계산 소요 시간
            double dt_solve = (ros::WallTime::now() - t0).toSec() - dt_gen;

            // 전체(총) 소요 시간
            double dt_total = (ros::WallTime::now() - t0).toSec();
            if (dt_total > max_total_time_)
            {
                max_gen_time_ = dt_gen;
                max_solve_time_ = dt_solve;
                max_total_time_ = dt_total;
            }

            optimal_path_pub_.publish(optimal_path);
            std_msgs::Float32 vmsg;
            std_msgs::Float32 amsg;
            vmsg.data = target_v;
            amsg.data = target_a;
            target_v_pub_.publish(vmsg);
            target_a_pub_.publish(amsg);

            // 최대값 로그 출력
            ROS_INFO("Max times [s]: gen = %.3f, solve = %.3f, total = %.3f",
                max_gen_time_, max_solve_time_, max_total_time_);
        }
        rate.sleep();
    }
}

// -------------------- Callback Implementations --------------------
void GlobalPathAnalyzer::GlobalPathCallback(const nav_msgs::Path::ConstPtr &msg)
{
    if (msg->poses.empty())
    {
        ROS_WARN("Received empty global_path");
        return;
    }
    if (!is_global_path_ready_)
    {
        vector<double> x_points, y_points;
        for (const auto &pose : msg->poses)
        {
            x_points.push_back(pose.pose.position.x);
            y_points.push_back(pose.pose.position.y);
        }
        s_vals_.clear();
        s_vals_.push_back(0.0);
        for (size_t i = 1; i < x_points.size(); i++)
        {
            double dist = hypot(x_points[i] - x_points[i-1], y_points[i] - y_points[i-1]);
            s_vals_.push_back(s_vals_.back() + dist);
        }
        total_length_ = s_vals_.back();

        if (s_candidates_.empty())
            {
                double step;
                s_candidates_.resize(num_samples_);
                step = total_length_ / (num_samples_ - 1);
                for (int i = 0; i < num_samples_; i++)
                    s_candidates_[i] = i * step;
            }
        // Create splines using tk::spline
        cs_x_.set_points(s_vals_, x_points);
        cs_y_.set_points(s_vals_, y_points);
        is_global_path_ready_ = true;
    }
}

void GlobalPathAnalyzer::odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
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
    if (!is_global_path_ready_ || s_candidates_.empty())
    {
        ROS_WARN("Global path not ready in obstacle callback");
        return;
    }

    // 이전 콜백에서 쌓인 장애물 정보 초기화
    sta_obs_.clear();
    dyn_obs_.clear();

    for (const auto &obstacle : msg->obstacle_list)
    {
        double x_obs = obstacle.position.x;
        double y_obs = obstacle.position.y;
        double s_obs, q_obs;
        compute_obstacle_frenet_all(x_obs, y_obs, s_obs, q_obs);
        if (s_obs >= s0_ && fabs(q_obs) <= 6.0)
        {
          sta_obs_.push_back({s_obs, q_obs, x_obs, y_obs, 0.0, 0.0});
        }
    }
    for (const auto &obstacle : msg->npc_list)
    {
        double x_obs = obstacle.position.x;
        double y_obs = obstacle.position.y;
        double s_obs, q_obs;
        compute_obstacle_frenet_all(x_obs, y_obs, s_obs, q_obs);
        if (fabs(q0_ - q_obs) <= 1.33 && s0_ > s_obs)
            continue;
        if (fabs(q_obs) <= 6.0)
        {
          dyn_obs_.push_back({s_obs, q_obs, x_obs, y_obs, obstacle.heading*M_PI/180.0, obstacle.velocity.x*10/36});
        }
    }
}

void GlobalPathAnalyzer::statusCallback(const morai_msgs::EgoVehicleStatus::ConstPtr &msg)
{
    current_speed_ = msg->velocity.x;
}

// -------------------- Utility Functions --------------------
double GlobalPathAnalyzer::FindClosestSNewton(double x0, double y0)
{
    if (!is_global_path_ready_)
        return 0.0;
    double s_current = s_candidates_[0];
    // Vectorized initial guess using candidate points:
    double min_dist = 1e12;
    for (size_t i = 0; i < s_candidates_.size(); i++) {
        double s_val = s_candidates_[i];
        double dx = x0 - cs_x_(s_val);
        double dy = y0 - cs_y_(s_val);
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
        double fprime = DistSqGrad(s_current, x0, y0);
        double fsecond = DistSqHess(s_current, x0, y0);
        if (fabs(fsecond) < 1e-12)
            break;
        double step = -fprime / fsecond;
        s_current += step;
        s_current = fmod(s_current, total_length_);
        if (s_current < 0) s_current += total_length_;
        if (fabs(step) < tol)
            break;
    }
    return s_current;
}

double GlobalPathAnalyzer::DistSqGrad(double s, double x0, double y0)
{
    double dx = x0 - cs_x_(s);
    double dy = y0 - cs_y_(s);
    double dxds = cs_x_.deriv(1, s);
    double dyds = cs_y_.deriv(1, s);
    return -2.0 * (dx * dxds + dy * dyds);
}

double GlobalPathAnalyzer::DistSqHess(double s, double x0, double y0)
{
    double dx = x0 - cs_x_(s);
    double dy = y0 - cs_y_(s);
    double dxds = cs_x_.deriv(1, s);
    double dyds = cs_y_.deriv(1, s);
    double d2xds2 = cs_x_.deriv(2, s);
    double d2yds2 = cs_y_.deriv(2, s);
    double val = (-dxds*dxds + dx * d2xds2) + (-dyds*dyds + dy * d2yds2);
    return -2.0 * val;
}

double GlobalPathAnalyzer::SignedLateralOffset(double x0, double y0, double s0)
{
    double x_s0 = cs_x_(s0);
    double y_s0 = cs_y_(s0);
    double dxds = cs_x_.deriv(1, s0);
    double dyds = cs_y_.deriv(1, s0);
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
    x_s = cs_x_(s);
    y_s = cs_y_(s);
    dxds = cs_x_.deriv(1, s);
    dyds = cs_y_.deriv(1, s);
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

void GlobalPathAnalyzer::compute_obstacle_frenet_all(double obs_x0, double obs_y0, double &obs_s0, double &obs_q0)
{
    obs_s0 = FindClosestSNewton(obs_x0, obs_y0);
    obs_q0 = SignedLateralOffset(obs_x0, obs_y0, obs_s0);
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
    for (const auto &obs : sta_obs_)
    {
        double dist = obs.s - s0;
        if (dist > 0 && dist <= 20)
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
    vector<pair<nav_msgs::Path,Pathinfo>>& candidate_paths,
    vector<pair<nav_msgs::Path,Pathinfo>>& cart_candidate_paths)
{
    candidate_paths.clear();
    cart_candidate_paths.clear();
    coll_info_on_paths_list.clear();

    // lane offsets 준비
    if(!lane_offsets_ready_)
    {
        for (double off = 5.0; off >= -1.0; off -= 0.1)
            lane_offsets_.push_back(off);
        lane_offsets_ready_ = true;
    }

    int N = static_cast<int>(lane_offsets_.size());
    candidate_paths.resize(N);
    cart_candidate_paths.resize(N);
    coll_info_on_paths_list.resize(N);

    // OpenMP 병렬 처리
    #pragma omp parallel for default(none) \
        shared(lane_offsets_, candidate_paths, cart_candidate_paths, s0, q0, N) \
        schedule(static)
    for (int i = 0; i < N; ++i) {
        generateLocalPath(s0, q0, lane_offsets_[i],
                          candidate_paths[i],
                          cart_candidate_paths[i],
                          i);
    }
}


void GlobalPathAnalyzer::generateLocalPath(double s0, double q0, double lane_offset,
    pair<nav_msgs::Path,Pathinfo> &path_msg, pair<nav_msgs::Path,Pathinfo> &cart_path_msg, int idx)
{
    path_msg.first.header.frame_id = "map";
    cart_path_msg.first.header.frame_id = "map";

    double delta_s = compute_delta_s_vel();
    double final_delta_s = compute_delta_s_with_obstacles(s0, delta_s);

    double dxds,dyds;
    double path_yaw;
    dxds = cs_x_.deriv(1, s0);
    dyds = cs_y_.deriv(1, s0);
    path_yaw = atan2(dyds, dxds);
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

    // s_min_ 내에서 장애물(정적, 동적)이 경로 상에 존재한다면 그 경로는 갈 수 없다고 판단하는 코드
    bool collision = false;
    double D = 0.0; // 경로 누적 길이
    double prev_x, prev_y;
    int index = 0;
    frenetToCartesian(s0, q0, prev_x, prev_y);
    for (double t : t_samples) {
        double q_val = eval_q_spline_t(a, b, c, d, t);
        double s_val;
        s_val = fmod(s0 + t, total_length_);

        if(!collision && t > 5.01/2)
        {
            double dqds  = 3*a*t*t + 2*b*t + c;
            double delta = atan(dqds);              // ego가 이 경로를 따라갔을 때 해당 위치에서의 헤딩 방향

            // (t 루프 안, delta 계산 직후)
            //const double ae = 1.167, be = 0.806;   // 타원 반장축/반단축(ego, 동적 장애물)
            const double ae = 3.54, be = 1.33;
            const double R  = 0.85;                // 정적 장애물 원 반지름

            // 타원(ego) 중심점
            double X0, Y0;
            frenetToCartesian(s_val, q_val, X0, Y0);

            // 경로 누적 길이 계산
            D += hypot(X0 - prev_x, Y0 - prev_y);
            prev_x = X0;
            prev_y = Y0;

            // 회전용 사인/코사인
            double cosd = cos(delta), sind = sin(delta);

            // 정적 장애물 중 하나라도 확장 타원 안에 들어오면 collision
            for (const auto &obs : sta_obs_)
            {
                if (0 <= obs.s - s0 && obs.s - s0 < 9.0)
                {
                    double dx = obs.x - X0;
                    double dy = obs.y - Y0;
                    // Frenet 회전 타원 좌표계로 변환
                    double x1 =  dx * cosd + dy * sind;
                    double y1 = -dx * sind + dy * cosd;
                    // 확장 타원 방정식 검사
                    double val = (x1*x1)/((ae + R)*(ae + R))
                                + (y1*y1)/((be + R)*(be + R));
                    if (val <= 1.0)
                    {
                        // 필요한 제동 가속도
                        double a_brake = -(current_speed_ * current_speed_) / (2.0 * D);
                        path_msg.second.target_v = 0.0;
                        path_msg.second.target_a = a_brake;
                        cart_path_msg.second.target_v = 0.0;
                        cart_path_msg.second.target_a = a_brake;
                        path_msg.second.possible      = false;
                        cart_path_msg.second.possible = false;
                        collision = true;
                        break;
                    }
                }
                else if(0 <= obs.s - s0 && obs.s - s0 < final_delta_s)
                {
                    double dx = obs.x - X0;
                    double dy = obs.y - Y0;
                    // Frenet 회전 타원 좌표계로 변환
                    double x1 =  dx * cosd + dy * sind;
                    double y1 = -dx * sind + dy * cosd;
                    // 확장 타원 방정식 검사
                    double val = (x1*x1)/((ae + R)*(ae + R))
                                + (y1*y1)/((be + R)*(be + R));
                    if (val <= 1.0)
                    {
                        // 필요한 제동 가속도
                        double a_brake = -(current_speed_ * current_speed_) / (2.0 * D);
                        path_msg.second.target_a = a_brake;
                        cart_path_msg.second.target_a = a_brake;

                        // 범위 벗어나면 불가능 처리
                        if (a_brake < a_min_)
                        {
                            path_msg.second.possible      = false;
                            cart_path_msg.second.possible = false;
                            path_msg.second.target_v = 0.0;
                            cart_path_msg.second.target_v = 0.0;
                            collision = true;
                            break;
                        }
                    }
                }
            }
            if(collision) // 충돌 지점 이후로 경로 생성 안함
                break;
            // 동적 장애물 중 하나라도 확장 타원 안에 들어오면 collision
            for (const auto &dyn : dyn_obs_)
            {
                double ds_obs = dyn.s - s0;
                double rel_vel = dyn.vel - current_speed_;
                const double b_obs = be;
                double a_obs = ae;
                if (ds_obs < -final_delta_s || ds_obs > s_min_)
                    continue;

                // 장애물 진행 각도 m
                double dyn_dtheta = normalize_angle(dyn.heading - path_yaw);
                double m = tan(dyn_dtheta);
                double cos_o = cos(dyn_dtheta), sin_o = sin(dyn_dtheta);

                // 프레네틱 상에서, 장애물 위치부터 final_delta_s 지점까지 N등분
                const int N_line = 20;
                double s_rel_start = ds_obs;
                double s_rel_end   = final_delta_s;
                double prev_X_line, prev_Y_line;
                frenetToCartesian(dyn.s, dyn.q, prev_X_line, prev_Y_line);
                double dyn_dist = 0.0;
                for (int j = 0; j <= N_line; ++j)
                {
                    double s_rel = s_rel_start + (s_rel_end - s_rel_start) * j / double(N_line);
                    // 직선 위의 q 값
                    double q_line = m * (s_rel - s_rel_start) + dyn.q;

                    // world 프레네틱 좌표
                    double s_world = fmod(s0 + s_rel, total_length_);
                    double X_line, Y_line;
                    frenetToCartesian(s_world, q_line, X_line, Y_line);
                    dyn_dist += hypot(X_line - prev_X_line, Y_line - prev_Y_line);
                    prev_X_line = X_line;
                    prev_Y_line = Y_line;
                    /*
                    ego의 거리를 알고 있으니 거리/현재속도 = 시간
                    동적 장애물의 이동거리를 구해서  거리/속도 = 시간
                    충돌이 발생했을 때 시간을 비교
                    */

                    // 타원 경계 샘플링
                    const int N = 30;
                    for (int n = 0; n < N; ++n)
                    {
                        double alpha = 2 * M_PI * n / N;
                        double x_loc = a_obs * cos(alpha);
                        double y_loc = b_obs * sin(alpha);

                        // 동적 장애물 타원 좌표 → 전역 좌표
                        double xd = X_line + x_loc * cos_o - y_loc * sin_o;
                        double yd = Y_line + x_loc * sin_o + y_loc * cos_o;

                        // ego 타원 좌표계로 변환
                        // ego 타원 중심·회전 정보는 이전에 계산된 X0,Y0, cosd, sind 사용
                        double dx = xd - X0, dy = yd - Y0;
                        double x1 =  dx * cosd + dy * sind;
                        double y1 = -dx * sind + dy * cosd;

                        // ego 타원 방정식 검사
                        double val = (x1*x1)/(ae*ae) + (y1*y1)/(be*be);
                        if (val <= 1.0)
                        {
                            coll_info_on_paths_list[idx].collpoint_on_path.push_back({
                                s_world,           // frenet s
                                q_line,            // frenet q
                                X_line,            // cart X
                                Y_line,            // cart Y
                                D,                 // (미리 계산해둔) 누적 거리
                                index
                            });
                            coll_info_on_paths_list[idx].dyn_obs.push_back(dyn);  // 전체 정보를 복사
                            coll_info_on_paths_list[idx].exist   = true;
                            index++;
                            break;
                        }
                    }
                }
            }
        }

        // Cartesian 변환
        double X, Y;
        frenetToCartesian(s_val, q_val, X, Y);

        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.pose.position.x = s_val;
        pose.pose.position.y = q_val;
        pose.pose.orientation.w = 1.0;
        path_msg.first.poses.push_back(pose);

        geometry_msgs::PoseStamped cart_pose;
        cart_pose.header.frame_id = "map";
        cart_pose.pose.position.x = X;
        cart_pose.pose.position.y = Y;
        cart_pose.pose.orientation.w = 1.0;
        cart_path_msg.first.poses.push_back(cart_pose);
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

// 카르다노 공식을 이용한 3차 방정식 해 함수
// At^3 + B t^2 + C t + D = 0 의 실근만 리턴
vector<double> GlobalPathAnalyzer::solveCubicReal(double A, double B, double C, double D)
{
    // 정규화: t^3 + a t^2 + b t + c = 0
    double a = B / A;
    double b = C / A;
    double c = D / A;

    double Q = (3*b - a*a) / 9.0;
    double R = (9*a*b - 27*c - 2*a*a*a) / 54.0;
    double disc = Q*Q*Q + R*R;

    vector<double> roots;
    if (disc >= 0) {
        // 실근 하나
        double sqrtD = sqrt(disc);
        double S = cbrt(R + sqrtD);
        double T = cbrt(R - sqrtD);
        roots.push_back(-a/3 + (S + T));
    } else {
        // 세 개의 실근
        double theta = acos(R / std::sqrt(-Q*Q*Q));
        double rho   = 2.0 * sqrt(-Q);
        roots.push_back(rho * cos(theta/3.0) - a/3.0);
        roots.push_back(rho * cos((theta + 2*M_PI)/3.0) - a/3.0);
        roots.push_back(rho * cos((theta + 4*M_PI)/3.0) - a/3.0);
    }
    return roots;
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
tuple<nav_msgs::Path,int,double,double> GlobalPathAnalyzer::computeOptimalPath(vector<pair<nav_msgs::Path,Pathinfo>>& candidate_paths)
{
    double w_s  = 1.0,
           w_sm = 10.0,
           w_g  = 10.0,
           w_d  = 1.0;

    const int N = candidate_paths.size();

    bool anyCollisionInfo = false;
    for (auto &ci : coll_info_on_paths_list) {
        if (ci.exist) { anyCollisionInfo = true; break; }
    }
    vector<double> cost_dynamic(N, 0.0);
    if (anyCollisionInfo) {
        cost_dynamic = computeDynamicCost(candidate_paths);
    }

    // feasible 인덱스만 모으기
    vector<int> valid_idxs;
    for (int i = 0; i < N; ++i) {
        if (candidate_paths[i].second.possible) {
            valid_idxs.push_back(i);
        }
    }

    bool useSubset = !valid_idxs.empty();

    // 비용 계산
    auto cost_static = computeStaticObstacleCost(candidate_paths, /*threshold=*/1.7);

    vector<double> cost_smooth(N, 0.0);
    if (useSubset) {
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < (int)valid_idxs.size(); ++k) {
            int i = valid_idxs[k];
            cost_smooth[i] = computeSmoothnessCostXY(candidate_paths[i].first);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            cost_smooth[i] = computeSmoothnessCostXY(candidate_paths[i].first);
        }
    }

    auto cost_global = computeGlobalPathCost(candidate_paths);

    // 유효 인덱스에 대해서만 total_cost 채우기
    vector<double> total_cost(N, numeric_limits<double>::infinity());
    if (useSubset)
    {
        for (int i : valid_idxs)
        {
            total_cost[i] =
                w_s  * cost_static[i] +
                w_sm * cost_smooth[i] +
                w_g  * cost_global[i] +
                w_d  * cost_dynamic[i];

            if(false)
            {
                ROS_INFO("Path %2zu | static=%6.3f | smooth=%6.3f | global=%6.3f | dynamic=%6.3f | total=%6.3f",
                         i,
                         w_s  * cost_static[i],
                         w_sm * cost_smooth[i],
                         w_g  * cost_global[i],
                         w_d  * cost_dynamic[i],
                         total_cost[i]);
            }
        }
    }
    else
    {
        for (int i = 0; i < N; ++i)
        {
            total_cost[i] =
                w_s  * cost_static[i] +
                w_sm * cost_smooth[i] +
                w_g  * cost_global[i] +
                w_d  * cost_dynamic[i];
        }
    }

    // 목표 속도와 목표 가속도 전경로 계산
    vector<double> v_targets(N,0.0), a_targets(N,0.0);
    if (useSubset)
    {
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < (int)valid_idxs.size(); ++k)
        {
            int i = valid_idxs[k];
            v_targets[i] = computeTargetVelocity(
                candidate_paths[i].first,
                5.56, 5.56, 5.0, 0.8,
                cost_static[i]
            );
            a_targets[i] = candidate_paths[i].second.target_a;
        }
    }
    else
    {
        // 모두 불가능 fallback -> 전체 계산
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
        {
            v_targets[i] = candidate_paths[i].second.target_v;
            a_targets[i] = candidate_paths[i].second.target_a;
        }
    }

    // 후보 선택
    int best_idx = -1;
    if (useSubset)
    {
        // 정상: total_cost 로 최솟값
        best_idx = valid_idxs[0];
        for (int i : valid_idxs) {
            if (total_cost[i] < total_cost[best_idx])
                best_idx = i;
        }
    }
    else
    {
        // 모두 불가능: target_a 가 가장 낮은(=가장 제동이 큰) 경로 선택
        ROS_WARN("All paths impossible: picking lowest-accel fallback");
        best_idx = 0;
        for (int i = 1; i < N; ++i) {
            if (a_targets[i] > a_targets[best_idx])
                best_idx = i;
        }
        /*
        // 모두 불가능: total_cost 가 가장 낮은 경로 선택
        ROS_WARN("All paths impossible: picking lowest-cost fallback");
        // total_cost 전체 중 최솟값의 인덱스 찾기
        best_idx = std::min_element(total_cost.begin(), total_cost.end())
                    - total_cost.begin();
        */
    }

    ROS_INFO("Optimal path: index:%2zu, static=%6.3f, smooth=%6.3f, global=%6.3f, dynamic=%6.3f, total=%6.3f, v:=%6.3f, a:=%6.3f",
        best_idx,
        w_s  * cost_static[best_idx],
        w_sm * cost_smooth[best_idx],
        w_g  * cost_global[best_idx],
        w_d  * cost_dynamic[best_idx],
        total_cost[best_idx],
        v_targets[best_idx],
        candidate_paths[best_idx].second.target_a
    );

    // 결과 반환
    nav_msgs::Path best_cart = convert_frenet_path_to_cartesian(candidate_paths[best_idx].first);
    return make_tuple(best_cart,
                      best_idx,
                      v_targets[best_idx],
                      a_targets[best_idx]);
}

// computeStaticObstacleCost 함수
vector<double> GlobalPathAnalyzer::computeStaticObstacleCost(
    vector<pair<nav_msgs::Path,Pathinfo>>& candidate_paths,
    double threshold)
{
    int N = static_cast<int>(candidate_paths.size());
    vector<double> indicators(N, 0.0);

    // 각 경로에 대해 인디케이터를 병렬 계산
    #pragma omp parallel for default(none) \
        shared(candidate_paths, sta_obs_, indicators, threshold, N)
    for (int pathnum = 0; pathnum < N; ++pathnum) {
        const auto &candidate_path = candidate_paths[pathnum].first;
        double indicator = 0.0;

        if (!sta_obs_.empty()) {
            // Frenet 좌표 (s, q) 상의 각 점과 모든 정적 장애물과의 최소 거리를 구함
            for (const auto &pose : candidate_path.poses) {
                double s_val = pose.pose.position.x;
                double q_val = pose.pose.position.y;

                double min_dist = numeric_limits<double>::infinity();
                for (const auto &obs : sta_obs_) {
                    double d = hypot(obs.s - s_val, obs.q - q_val);
                    if (d < min_dist) min_dist = d;
                }
                // 최소 거리가 임계값보다 작으면 인디케이터를 1로 설정
                if (min_dist < threshold) {
                    indicator = 1.0;
                    break;
                }
            }
        }

        indicators[pathnum] = indicator;
    }

    // Gaussian smoothing 은 직렬로 수행
    vector<double> cost_profile = gaussian_filter1d(indicators, 10.0);
    return cost_profile;
}

// 1차원 가우시안 필터 함수
vector<double> GlobalPathAnalyzer::gaussian_filter1d(const vector<double>& input, double sigma)
{
    int n = input.size();
    vector<double> output(n, 0.0);
    // 커널 반경: 보통 3*sigma 정도를 사용 (이보다 더 큰 값은 거의 0에 수렴)
    int radius = ceil(3 * sigma);
    int kernel_size = 2 * radius + 1;
    vector<double> kernel(kernel_size, 0.0);
    double sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        double val = exp(- (i * i) / (2 * sigma * sigma));
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
            idx = max(0, min(idx, n - 1));
            conv += input[idx] * kernel[j + radius];
        }
        output[i] = conv;
    }
    return output;
}

// computeSmoothnessCostXY 함수
double GlobalPathAnalyzer::computeSmoothnessCostXY(const nav_msgs::Path &frenet_path)
{
    vector<pair<double, double>> xy_points;
    for (const auto &pose : frenet_path.poses) {
        double s_val = pose.pose.position.x;
        double q_val = pose.pose.position.y;
        double X, Y;
        frenetToCartesian(s_val, q_val, X, Y);
        xy_points.push_back(make_pair(X, Y));
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

vector<double> GlobalPathAnalyzer::computeGlobalPathCost(
    vector<pair<nav_msgs::Path,Pathinfo>>& candidate_paths)
{
    const int N = candidate_paths.size();
    vector<double> offsets(N, 0.0);

    // 1) offsets 계산 병렬화
    #pragma omp parallel for default(none) shared(candidate_paths, offsets, N)
    for (int i = 0; i < N; ++i) {
        double sum_q = 0.0;
        const auto& path = candidate_paths[i].first;
        for (const auto &pose : path.poses) {
            sum_q += fabs(pose.pose.position.y);
        }
        offsets[i] = sum_q;
    }

    // 2) total_offset 계산 (reduction)
    double total_offset = 0.0;
    #pragma omp parallel for reduction(+:total_offset) default(none) shared(offsets, N)
    for (int i = 0; i < N; ++i) {
        total_offset += offsets[i];
    }

    vector<double> cg_list(N, 0.0);
    if (total_offset < 1e-9) {
        // 모두 0으로 초기화된 cg_list 반환
        return cg_list;
    }

    // 3) cg_list 계산 병렬화
    #pragma omp parallel for default(none) shared(offsets, cg_list, total_offset, N)
    for (int i = 0; i < N; ++i) {
        cg_list[i] = offsets[i] / total_offset;
    }

    return cg_list;
}

vector<double> GlobalPathAnalyzer::computeDynamicCost(
    vector<pair<nav_msgs::Path,Pathinfo>>& candidate_paths)
{
    const int N = static_cast<int>(candidate_paths.size());
    vector<double> dynamic_costs(N, 0.0);

    const double L_cut_in = 5.0;   // 끼어들기에 필요한 추가 거리 (m)
    const double L0       = 3.0;   // 기본 추종 거리 (m)
    const double v_veh    = current_speed_;  // 최소 1m/s

    //병렬 for문
    #pragma omp parallel for default(none) \
        shared(candidate_paths, coll_info_on_paths_list, dyn_obs_, dynamic_costs, v_veh, L_cut_in, L0, N, a_min_) \
        schedule(static)
    for (int i = 0; i < N; ++i)
    {
        const auto &ci = coll_info_on_paths_list[i];
        double final_cost = 0.0;
        double final_a_req = 0.0;
        double final_s_c = 0.0;
        double final_t_obs = 0.0;
        double final_delta_t = 0.0;
        bool first_cost = false;
        if (ci.exist)
        {
            // 이 경로에 대해 등록된 모든 충돌 후보 지점 순회
            for (const auto &coll : ci.collpoint_on_path)
            {
                // 충돌 거리
                double s_c   = coll.dist;
                double t_veh = max(s_c / v_veh, 0.05);

                // 충돌 지점에서의 동적 장애물 정보
                const auto &obs = ci.dyn_obs[coll.index];
                double dx       = coll.x - obs.x;
                double dy       = coll.y - obs.y;
                double d        = hypot(dx, dy);
                double v_o      = obs.vel; // max(obs.vel, 1.0);
                double t_obs    = max(d / v_o, 0.05);

                double delta_t = t_obs - t_veh;
                double a_req   = 0.0;
                double cost = 0.0;
                if (delta_t > 0.0)
                {
                    // Cut-in 시나리오
                    double delta_cut = s_c + L_cut_in - v_veh * t_obs;
                    if (delta_cut > 0.0)
                        a_req = 2.0 * delta_cut / (t_obs * t_obs);
                    cost = fabs(a_req) * (s_c + L_cut_in);
                    // a_req = max(a_min_, a_req);
                }
                else
                {
                    // Follow 시나리오
                    double Lf = (L0 <= s_c) ? L0 : s_c;
                    double delta_f = s_c - Lf - v_veh * t_obs;
                    a_req = 2.0 * delta_f / (t_obs * t_obs);
                    cost = fabs(a_req) * (s_c - L0);
                    a_req = min(a_min_, a_req);
                }

                if (!first_cost)
                {
                    first_cost = true;
                    final_cost = cost;
                    final_a_req = a_req;
                    final_s_c = s_c;
                    final_t_obs = t_obs;
                    final_delta_t = delta_t;
                }
                else if (final_cost < cost)
                {
                    final_cost = cost;
                    final_a_req = a_req;
                    final_s_c = s_c;
                    final_t_obs = t_obs;
                    final_delta_t = delta_t;
                }
            }

            candidate_paths[i].second.target_a = final_a_req;
            // 가속/제동 한계 위반 체크
            if (final_a_req < a_min_ || final_a_req > a_max_)
            {
                candidate_paths[i].second.possible = false;
                candidate_paths[i].second.target_v = 0.0;
            }
            else if (final_a_req > 0 && current_speed_ >= 5.3) // 현재 최대 속도인데 요구되는 가속도가 양수일 때
            {
                candidate_paths[i].second.possible = false;
                candidate_paths[i].second.target_v = 5.56;
                candidate_paths[i].second.target_a = 0.0;
            }
            dynamic_costs[i] = final_cost;
        }
    }
    return dynamic_costs;
}

double GlobalPathAnalyzer::computeTargetVelocity(
    const nav_msgs::Path &path,
    double v_limit,
    double v_ref,
    double a_lat_max,
    double k_s,
    double C_s)
{
    // 1) Cartesian 좌표로 변환해서 (x,y) 리스트 구성
    vector<pair<double,double>> xy;
    xy.reserve(path.poses.size());
    for (const auto &pose : path.poses) {
        double s = pose.pose.position.x;
        double q = pose.pose.position.y;
        double X,Y;
        frenetToCartesian(s, q, X, Y);
        xy.emplace_back(X,Y);
    }

    // 2) 곡률 κ(s) 계산 & 최대값 찾기
    double max_kappa = 0.0;
    // i=1..N-2 까지 세 점씩 curvature 근사
    for (size_t i = 1; i+1 < xy.size(); ++i)
    {
        double k = approxCurvature(
            xy[i-1].first, xy[i-1].second,
            xy[i  ].first, xy[i  ].second,
            xy[i+1].first, xy[i+1].second
        );
        max_kappa = max(max_kappa, abs(k));
    }
    // 논문 식 (17)
    double v_k = (max_kappa > 1e-9)
            ? sqrt(a_lat_max / max_kappa)
            : v_limit;  // 거의 직선이면 제한속도로

    // 논문 식 (18)
    double v_s = (1.0 - k_s * C_s * C_s) * v_ref;
    v_s = max(0.0, v_s);

    // 4) 최종 목표 속도는 가장 느린 것
    return min({ v_limit, v_k, v_s });
}

double GlobalPathAnalyzer::computeMaxCurvature(const nav_msgs::Path &path)
{
    const auto &pts = path.poses;
    if (pts.size() < 3) return 0.0;

    double max_kappa = 0.0;
    for (size_t i = 1; i + 1 < pts.size(); ++i)
    {
        double x_prev = pts[i-1].pose.position.x;
        double y_prev = pts[i-1].pose.position.y;
        double x_curr = pts[i  ].pose.position.x;
        double y_curr = pts[i  ].pose.position.y;
        double x_next = pts[i+1].pose.position.x;
        double y_next = pts[i+1].pose.position.y;

        double kappa = approxCurvature(
            x_prev, y_prev,
            x_curr, y_curr,
            x_next, y_next
        );
        max_kappa = std::max(max_kappa, std::abs(kappa));
    }
    return max_kappa;
}

// -------------------- Main --------------------
int main(int argc, char** argv) {
    omp_set_num_threads(4);
    ros::init(argc, argv, "local_path_pub");
    ros::NodeHandle nh;
    GlobalPathAnalyzer analyzer(nh);
    analyzer.spin();
    return 0;
}

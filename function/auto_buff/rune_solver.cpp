#include <atomic>
#include <mutex>

#include "rune_solver.hpp"
#include "rune_detector.hpp"

namespace auto_buff
{

  std::atomic<bool> STOP_THREAD(false);
  std::atomic<bool> VALID_PARAMS(false);
  extern std::mutex MUTEX;

  /**
   * @brief 初始化解算参数
   * @param[in]
   */
  BuffSolver::BuffSolver(const std::string& config)
  {
    auto yaml = YAML::LoadFile(config);

    FPS = yaml["FPS"].as<int>();
    min_fit_data_size = yaml["min_fit_data_size"].as<int>();
    max_fit_data_size = yaml["max_fit_data_size"].as<int>();

    min_bullet_speed = yaml["min_bullet_speed"].as<double>();
    default_bullet_speed = yaml["default_bullet_speed"].as<double>();
    compansate_time = yaml["compansate_time"].as<double>();
    compansate_pitch = yaml["compansate_pitch"].as<double>();
    compansate_yaw = yaml["compansate_yaw"].as<double>();
  }

  BuffSolver::~BuffSolver()
  {
    STOP_THREAD.store(true);
    m_fit_thread.join();
  };

  // Construct a new Calculator:: Calculator object
  void BuffSolver::init_params()
  {
    // ================= 世界坐标点（能量机关几何模型） =================
    m_world_points = {
        {-0.5f * BuffDetection::armor_inside_width, BuffDetection::armor_inside_y, 0.0f},

        {0.5f * BuffDetection::armor_inside_width, BuffDetection::armor_inside_y, 0.0f},

        {0.0f, -BuffDetection::armor_outside_y - BuffDetection::armor_outside_height, 0.0f},

        {-0.5f * BuffDetection::armor_outside_width, -BuffDetection::armor_outside_y, 0.0f},

        {0.5f * BuffDetection::armor_outside_width, -BuffDetection::armor_outside_y, 0.0f},

        {0.0f, BuffDetection::power_rune_radius, 0.0f}};

    // ================= 状态量初始化 =================
    m_direction = auto_buff::Direction::UNKNOWN;
    m_convexity = auto_buff::Convexity::UNKNOWN;
    m_total_shift = 0;
    m_first_detect = true;
    m_angle_last = 0.0;

    // ================= 方向判断阈值 =================
    const int interval_ms = 1000 / FPS;
    m_direction_thresh = std::max(100 / interval_ms, 2);

    // ================= 拟合数据 & 线程 =================
    m_fit_data.reserve(FPS * 20);
    m_fit_thread = std::thread(&BuffSolver::fit, this);
  }

  /**
   * @brief 解算，分别进行预处理，矩阵解算，设置第一次检测，角度解算，旋转方向解算和预测
   */
  bool BuffSolver::calculate(const Frame& frame, std::vector<cv::Point2f>& cameraPoints)
  {
    preprocess(frame, cameraPoints);
    if (matrix_cal() == false) {
      return false;
    }
    set_first_detect();
    angle_cal();
    directionCal();
    if (m_direction == Direction::UNKNOWN) {
      return false;
    }
    if (predict() == false) {
      return false;
    }
    return true;
  }

  /**
   * @brief 设置第一次检测的旋转矩阵和时间戳，便于进行角度解算
   */
  void BuffSolver::set_first_detect()
  {
    if (m_first_detect) {
      m_first_detect = false;
      m_rMatW2RBase = m_rMatW2R.clone();
      m_start_time = m_frame_time;
    }
  }

  /**
   * @brief 角度解算
   */
  void BuffSolver::angle_cal()
  {
    // 计算相对于第一次检测旋转的角度 angelAbs
    cv::Mat rMatRel{m_rMatW2RBase.inv() * m_rMatW2R};
    double angleAbs{-std::atan2(rMatRel.at<double>(0, 1), rMatRel.at<double>(0, 0))};
    // 减去上一次得到的角度，得到相对于上一次检测旋转的角度
    double angleMinus{angleAbs - m_angle_last};
    m_angle_last = angleAbs;
    // 用这个角度除以两片扇叶的夹角，得到装甲板切换数，并计算总的装甲板切换数
    int shift = std::round(angleMinus / angle_between_fan_blades);
    m_total_shift += shift;
    // 用 angelAbs 减去装甲板切换的角度，即可得到连续的角度
    m_angle_rel = angleAbs - m_total_shift * angle_between_fan_blades;
    double time{
        std::chrono::duration_cast<std::chrono::microseconds>(m_frame_time - m_start_time).count() /
        1e6};
    // 存储相对于第一次识别的时间间隔和角度的绝对值，日后进行拟合
    if (BuffDetection::MODE == Mode::BIG) {
      std::unique_lock lock(m_mutex);
      m_fit_data.emplace_back(time, std::abs(m_angle_rel));
    }
  }

  /**
   * @brief 拟合线程的主函数
   */
  void BuffSolver::fit()
  {
    if (BuffDetection::MODE != Mode::BIG) {
      return;
    }

    decltype(m_fit_data) fitData;
    while (STOP_THREAD.load() == false) {
      {
        std::shared_lock lock(m_mutex);
        fitData = m_fit_data;
      }
      // 数据量过少时，直接返回
      if (m_fit_data.size() < (size_t)min_fit_data_size) {
        continue;
      }
      bool result = fit_once();
      VALID_PARAMS.store(result);
#if CONSOLE_OUTPUT >= 1
      MUTEX.lock();
      if (result == true) {
        std::cout << "params: ";
        std::for_each(m_params.begin(), m_params.end(), [](auto&& it) { std::cout << it << " "; });
        std::cout << std::endl;
      }
      MUTEX.unlock();
#endif
      if (m_fit_data.size() > (size_t)max_fit_data_size) {
        m_fit_data.erase(m_fit_data.begin(), m_fit_data.begin() + m_fit_data.size() / 2);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(int(1e4 / FPS)));
    }
  }

  /**
   * @brief 拟合一次
   */
  bool BuffSolver::fit_once()
  {
    // 如果数据量过少，则确定凹凸性
    if (m_fit_data.size() < (size_t)2 * min_fit_data_size) {
      m_convexity = get_convexity(m_fit_data);
    }
    // 利用 ransac 算法计算参数
    m_params = ransac_fitting(m_fit_data, m_convexity);
    return true;
  }

  /**
   * @brief 凹凸性计算
   * @param[in] data          角度数据
   * @return Convexity
   */
  Convexity get_convexity(const std::vector<std::pair<double, double>>& data)
  {
    auto first{data.begin()}, last{data.end() - 1};
    double slope{(last->second - first->second) / (last->first - first->first)};
    double offset{(first->second * last->first - last->second * first->first) /
                  (last->first - first->first)};
    int concave{0}, convex{0};
    for (const auto& i : data) {
      if (slope * i.first + offset > i.second) {
        concave++;
      } else {
        convex++;
      }
    }
    const int standard{static_cast<int>(data.size() * 0.75)};
    return concave > standard  ? Convexity::CONCAVE
           : convex > standard ? Convexity::CONVEX
                               : Convexity::UNKNOWN;
  }

  /**
   * @brief ransac 算法，返回拟合参数
   * @param[in] data          角度数据
   * @param[in] convexity     凹凸性
   * @return std::array<double, 5>
   */
  std::array<double, 5> ransac_fitting(const std::vector<std::pair<double, double>>& data,
                                       Convexity convexity)
  {
    // inliers 为符合要求的点，outliers 为不符合要求的点
    std::vector<std::pair<double, double>> inliers, outliers;
    // 初始时，inliers 为全部点
    inliers.assign(data.begin(), data.end());
    // 迭代次数
    int iterTimes{data.size() < 400 ? 200 : 20};
    // 初始参数
    std::array<double, 5> params{0.470, 1.942, 0, 1.178, 0};
    for (int i = 0; i < iterTimes; ++i) {
      decltype(inliers) sample;
      // 如果数据点较多，则将数据打乱，取其中一部分
      if (inliers.size() > 400) {
        std::shuffle(inliers.begin(), inliers.end() - 100,
                     std::default_random_engine(
                         std::chrono::system_clock::now().time_since_epoch().count()));
        sample.assign(inliers.end() - 200, inliers.end());
      } else {
        sample.assign(inliers.begin(), inliers.end());
      }
      // 进行拟合
      params = least_square_estimate(sample, params, convexity);
      // 对 inliers 每一个计算误差
      std::vector<double> errors;
      for (const auto& inlier : inliers) {
        errors.push_back(std::abs(inlier.second - tools::get_angle_big(inlier.first, params)));
      }
      // 如果数据量较大，则对点进行筛选
      if (data.size() > 800) {
        std::sort(errors.begin(), errors.end());
        const int index{static_cast<int>(errors.size() * 0.95)};
        const double threshold{errors[index]};
        // 剔除 inliers 中不符合要求的点
        for (size_t i = 0; i < inliers.size() - 100; ++i) {
          if (std::abs(inliers[i].second - tools::get_angle_big(inliers[i].first, params)) >
              threshold) {
            outliers.push_back(inliers[i]);
            inliers.erase(inliers.begin() + i);
          }
        }
        // 将 outliers 中符合要求的点加进来
        for (size_t i = 0; i < outliers.size(); ++i) {
          if (std::abs(outliers[i].second - tools::get_angle_big(outliers[i].first, params)) <
              threshold) {
            inliers.emplace(inliers.begin(), outliers[i]);
            outliers.erase(outliers.begin() + i);
          }
        }
      }
    }
    // 返回之前对所有 inliers 再拟合一次
    return params;
  }

  /**
   * @brief 最小二乘拟合，返回参数列表
   * @param[in] points        数据点
   * @param[in] params        初始参数
   * @param[in] convexity     凹凸性
   * @return std::array<double, 5>
   */
  std::array<double, 5> least_square_estimate(const std::vector<std::pair<double, double>>& points,
                                              const std::array<double, 5>& params,
                                              Convexity convexity)
  {
    std::array<double, 5> ret = params;
    ceres::Problem problem;
    for (size_t i = 0; i < points.size(); i++) {
      ceres::CostFunction* costFunction = new CostFunctor2(points[i].first, points[i].second);
      ceres::LossFunction* lossFunction = new ceres::SoftLOneLoss(0.1);
      problem.AddResidualBlock(costFunction, lossFunction, ret.begin());
    }
    std::array<double, 3> omega;
    if (points.size() < 100) {
      // 在数据量较小时，可以利用凹凸性定参数边界
      if (convexity == Convexity::CONCAVE) {
        problem.SetParameterUpperBound(ret.begin(), 2, -2.8);
        problem.SetParameterLowerBound(ret.begin(), 2, -4);
      } else if (convexity == Convexity::CONVEX) {
        problem.SetParameterUpperBound(ret.begin(), 2, -1.1);
        problem.SetParameterLowerBound(ret.begin(), 2, -2.3);
      }
      omega = {10., 1., 1.};
    } else {
      // 而数据量较多后，则不再需要凹凸性辅助拟合
      omega = {60., 50., 50.};
    }
    ceres::CostFunction* costFunction1 = new CostFunctor1(ret[0], 0);
    ceres::LossFunction* lossFunction1 =
        new ceres::ScaledLoss(new ceres::HuberLoss(0.1), omega[0], ceres::TAKE_OWNERSHIP);
    problem.AddResidualBlock(costFunction1, lossFunction1, ret.begin());
    ceres::CostFunction* costFunction2 = new CostFunctor1(ret[1], 1);
    ceres::LossFunction* lossFunction2 =
        new ceres::ScaledLoss(new ceres::HuberLoss(0.1), omega[1], ceres::TAKE_OWNERSHIP);
    problem.AddResidualBlock(costFunction2, lossFunction2, ret.begin());
    ceres::CostFunction* costFunction3 = new CostFunctor1(ret[3], 3);
    ceres::LossFunction* lossFunction3 =
        new ceres::ScaledLoss(new ceres::HuberLoss(0.1), omega[2], ceres::TAKE_OWNERSHIP);
    problem.AddResidualBlock(costFunction3, lossFunction3, ret.begin());
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = false;
    options.check_gradients = false;
    options.gradient_check_relative_precision = 1e-4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    return ret;
  }

  /**
   * @brief 预处理，更新时间戳并设置弹速
   */
  void BuffSolver::preprocess(const Frame& frame, std::vector<cv::Point2f>& cameraPoints)
  {
    m_camera_points = cameraPoints;
    m_frame_time = frame.m_time;
    m_receive_roll = frame.m_roll;
    m_receive_pitch = frame.m_pitch;
    m_receive_yaw = frame.m_yaw;
    m_bullet_speed =
        current_bullet_speed > min_bullet_speed ? current_bullet_speed : default_bullet_speed;
  }

  /**
   * @brief
   * 矩阵解算，分别进行世界坐标系到相机坐标系，相机坐标系到云台坐标系，云台坐标系到机器人坐标系的转换，与旋转矩阵的设置
   */
  bool BuffSolver::matrix_cal()
  {
    // 进行坐标变换并设置旋转矩阵
    m_matW2C = world2Camera(m_world_points, m_camera_points, intrinsic_matrix, dist_coeffs);
    m_matC2G = camera2Gimbal(camera2gimbal_translation, camera2gimbal_rotation);
    m_matG2R =
        gimbal2Robot(tools::angle2Radian(m_receive_pitch), tools::angle2Radian(m_receive_yaw),
                     tools::angle2Radian(m_receive_roll));
    m_matW2R = m_matG2R * m_matC2G * m_matW2C;
    m_rMatW2R = m_matW2R(cv::Rect(0, 0, 3, 3));
    m_distance2target = cv::norm(m_matW2C.col(3)) * 1e-3;
    if (tools::inRange<double>(m_distance2target, min_distance_to_target, max_distance_to_target) ==
        false) {
      return false;
    }
    // 记录装甲板和中心 R 的机器人坐标
    cv::Mat armorWorld = (cv::Mat_<double>(4, 1) << 0, 0, 0, 1);
    cv::Mat centerWorld = (cv::Mat_<double>(4, 1) << 0, BuffDetection::power_rune_radius, 0, 1);
    cv::Mat armorRobot{m_matW2R * armorWorld};
    cv::Mat centerCamera{m_matW2C * centerWorld};
    cv::Mat centerRobot{m_matW2R * centerWorld};
    m_armor_robot = {(float)(armorRobot.at<double>(0, 0)), (float)(armorRobot.at<double>(1, 0)),
                     (float)(armorRobot.at<double>(2, 0))};
    m_center_robot = {(float)(centerRobot.at<double>(0, 0)), (float)(centerRobot.at<double>(1, 0)),
                      (float)(centerRobot.at<double>(2, 0))};
#if CONSOLE_OUTPUT >= 2
    MUTEX.lock();
    std::cout << "armor center coordinate: " << m_armor_robot << std::endl;
    std::cout << "center R coordinate: " << m_center_robot << std::endl;
    MUTEX.unlock();
#endif
    return true;
  }

  /**
   * @brief 世界坐标系转相机坐标系
   * @param[in] worldPoints   世界坐标系坐标
   * @param[in] cameraPoints  相机坐标系坐标
   * @return cv::Mat
   */
  cv::Mat world2Camera(const std::vector<cv::Point3f>& worldPoints,
                       const std::vector<cv::Point2f>& cameraPoints, const cv::Mat& intrinsicMatrix,
                       const cv::Mat& distCoeffs)
  {
    cv::Mat rVec, tVec, rMat;
    cv::solvePnP(worldPoints, cameraPoints, intrinsicMatrix, distCoeffs, rVec, tVec, false,
                 cv::SOLVEPNP_ITERATIVE);
    cv::Rodrigues(rVec, rMat);
    cv::Mat w2c{cv::Mat::zeros(cv::Size(4, 4), CV_64FC1)};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        w2c.at<double>(i, j) = rMat.at<double>(i, j);
      }
    }
    for (int i = 0; i < 3; ++i) {
      w2c.at<double>(i, 3) = tVec.at<double>(i, 0);
    }
    w2c.at<double>(3, 3) = 1.0;
    return w2c;
  }

  /**
   * @brief 相机坐标系转云台坐标系
   * @param[in] r             旋转参数
   * @param[in] t             平移参数
   * @return cv::Mat
   */
  cv::Mat camera2Gimbal(const std::array<double, 3>& r, const std::array<double, 3>& t)
  {
    cv::Mat rVec = (cv::Mat_<double>(3, 1) << r[0], r[1], r[2]);
    cv::Mat tVec = (cv::Mat_<double>(3, 1) << t[0], t[1], t[2]);
    cv::Mat rMat;
    cv::Rodrigues(rVec, rMat);
    cv::Mat c2g{cv::Mat::zeros(cv::Size(4, 4), CV_64FC1)};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        c2g.at<double>(i, j) = rMat.at<double>(i, j);
      }
    }
    for (int i = 0; i < 3; ++i) {
      c2g.at<double>(i, 3) = tVec.at<double>(i, 0);
    }
    c2g.at<double>(3, 3) = 1;
    return c2g;
  }

  /**
   * @brief 云台坐标系转机器人坐标系
   * @param[in] pitch
   * @param[in] yaw
   * @return cv::Mat
   */
  cv::Mat gimbal2Robot(double pitch, double yaw, double roll)
  {
    cv::Mat matY = (cv::Mat_<double>(4, 4) << std::cos(-yaw), 0, std::sin(-yaw), 0, 0, 1, 0, 0,
                    -std::sin(-yaw), 0, std::cos(-yaw), 0, 0, 0, 0, 1);
    cv::Mat matX = (cv::Mat_<double>(4, 4) << 1, 0, 0, 0, 0, std::cos(pitch), -std::sin(pitch), 0,
                    0, std::sin(pitch), std::cos(pitch), 0, 0, 0, 0, 1);
    cv::Mat matZ = (cv::Mat_<double>(4, 4) << std::cos(roll), -std::sin(roll), 0, 0, std::sin(roll),
                    std::cos(roll), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    // 实际测试中，roll 不起作用，这可能和电控对欧拉角的解算有关
    return matY * matX;
  }

  /**
   * @brief 旋转方向解算
   */
  void BuffSolver::directionCal()
  {
    if (m_direction == Direction::UNKNOWN || m_direction == Direction::STABLE) {
      m_direction_data.push_back(m_angleRel);
      if ((int)m_direction_data.size() >= m_direction_thresh) {
        // 计算角度差并投票
        int stable = 0, anti = 0, clockwise = 0;
        for (size_t i = 0; i < m_direction_data.size() / 2; ++i) {
          auto temp{m_direction_data.at(i + m_direction_data.size() / 2) - m_direction_data.at(i)};
          if (temp > +1.5e-2) {
            clockwise++;
          } else if (temp < -1.5e-2) {
            anti++;
          } else {
            stable++;
          }
        }
        // 得票数最多的为对应旋转方向
        if (int temp{std::max({stable, clockwise, anti})}; temp == clockwise) {
          m_direction = Direction::CLOCKWISE;
        } else if (temp == anti) {
          m_direction = Direction::ANTI_CLOCKWISE;
        } else {
          m_direction = Direction::STABLE;
        }
      }
    }
#if CONSOLE_OUTPUT >= 2
    MUTEX.lock();
    std::cout << "direction: "
              << (m_direction == Direction::CLOCKWISE        ? "clockwise"
                  : m_direction == Direction::ANTI_CLOCKWISE ? "anti-clockwise"
                  : m_direction == Direction::STABLE         ? "stable"
                                                             : "unknown")
              << std::endl;
    MUTEX.unlock();
#endif
  }

  /**
   * @brief 预测
   */
  bool BuffSolver::predict()
  {
    double angle;
    if (m_direction == Direction::STABLE) {
      angle = 0.0;
    } else {
      if (BuffDetection::MODE == Mode::BIG) {
        auto frameTime{
            std::chrono::duration_cast<std::chrono::milliseconds>(m_frame_time - m_start_time)
                .count()};
        if (VALID_PARAMS.load() == false) {
          return false;
        }
        angle = tools::get_rotation_angle_big(m_distance2target, m_bullet_speed, m_params,
                                              BuffSolver::compansate_time, frameTime);
      } else {
        angle = tools::get_rotation_angle_small(m_distance2target, m_bullet_speed,
                                                BuffSolver::small_power_rune_rotation_speed,
                                                BuffSolver::compansate_time);
      }
      if (m_direction == Direction::ANTI_CLOCKWISE) {
        angle = -angle;
      }
    }
    cv::Mat matrixWorld =
        (cv::Mat_<double>(4, 1) << BuffDetection::power_rune_radius * std::sin(angle),
         BuffDetection::power_rune_radius - BuffDetection::power_rune_radius * std::cos(angle), 0.0,
         1.0);
    cv::Mat matrixRobot{m_matW2R * matrixWorld};
    m_predict_robot = {(float)(matrixRobot.at<double>(0, 0)), (float)(matrixRobot.at<double>(1, 0)),
                       (float)(matrixRobot.at<double>(2, 0))};
    auto [predictPitch, predictYaw] = getPitchYawFromRobotCoor(m_predict_robot, m_bullet_speed);
    m_predict_pitch = predictPitch;
    m_predict_yaw = predictYaw;
    m_predict_pixel = getPixelFromRobot(m_predict_robot, m_matW2C, m_matW2R);
#if CONSOLE_OUTPUT >= 2
    MUTEX.lock();
    std::cout << "predict angle: " << angle << std::endl;
    std::cout << "predict coordinate: " << m_predict_robot << std::endl;
    std::cout << "predict pitch and yaw: " << m_predict_pitch << ", " << m_predict_yaw << std::endl;
    MUTEX.unlock();
#endif
    return true;
  }

  cv::Point2f BuffSolver::getPixelFromCamera(const cv::Mat& intrinsicMatrix,
                                             const cv::Mat& cameraPoint)
  {
    double fx = intrinsicMatrix.at<double>(0, 0);
    double fy = intrinsicMatrix.at<double>(1, 1);
    double cx = intrinsicMatrix.at<double>(0, 2);
    double cy = intrinsicMatrix.at<double>(1, 2);
    double X = cameraPoint.at<double>(0, 0);
    double Y = cameraPoint.at<double>(1, 0);
    double Z = cameraPoint.at<double>(2, 0);
    double u = (fx * X + cx * Z) / Z;
    double v = (fy * Y + cy * Z) / Z;
    return cv::Point2f(u, v);
  }

  cv::Point2f BuffSolver::getPixelFromRobot(const cv::Point3f& robot, const cv::Mat& w2c,
                                            const cv::Mat& w2r)
  {
    cv::Mat matrixRobotPoint = (cv::Mat_<double>(4, 1) << robot.x, robot.y, robot.z, 1.0);
    cv::Mat matrixCameraPoint{w2c * (w2r.inv() * matrixRobotPoint)};
    return getPixelFromCamera(BuffSolver::intrinsic_matrix, matrixCameraPoint);
  }

  std::pair<double, double> BuffSolver::getPitchYawFromRobotCoor(const cv::Point3f& target,
                                                                 double bulletSpeed)
  {
    double horizontal{tools::p2p_distance({0.0, 0.0}, {target.x, target.z}) * 1e-3};
    double a{-0.5 * BuffSolver::gravity * std::pow(horizontal, 2) / std::pow(bulletSpeed, 2)};
    double b{horizontal};
    double c{a + target.y * 1e-3};
    double result{tools::solveQuadraticEquation(a, b, c).second};
    double pitch{tools::radian2Angle(std::atan(result)) + BuffSolver::compansate_pitch};
    double yaw{tools::radian2Angle(-std::atan2(target.x, target.z)) + BuffSolver::compansate_yaw};
    return std::make_pair(pitch, yaw);
  }

} // namespace auto_buff
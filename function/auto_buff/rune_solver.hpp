#pragma once

#include <ceres/ceres.h>

#include "rune_detector.hpp"

#include <yaml-cpp/yaml.h>
#include <tools/logger.hpp>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <random>
#include <shared_mutex>
#include <thread>

namespace auto_buff
{
  class BuffSolver
  {
  public:
    BuffSolver(const std::string& config);
    ~BuffSolver();

    bool calculate(const Frame& frame, std::vector<cv::Point2f>& cameraPoints);

    void init_params();

    static int get_fps() { return FPS; };
    inline cv::Point3f getPredictRobot() { return m_predict_robot; }
    inline cv::Point2f getPredictPixel() { return m_predict_pixel; }

  private:
    // ================= 方法 =================
    void fit();
    bool fit_once();
    void preprocess(const Frame& frame, std::vector<cv::Point2f>& cameraPoints);
    bool matrix_cal();
    void set_first_detect();
    void angle_cal();
    void directionCal();
    bool predict();
    cv::Point2f getPixelFromCamera(const cv::Mat& intrinsicMatrix, const cv::Mat& cameraPoint);
    cv::Point2f getPixelFromRobot(const cv::Point3f& robot, const cv::Mat& w2c, const cv::Mat& w2r);
    std::pair<double, double> getPitchYawFromRobotCoor(const cv::Point3f& target,
                                                       double bulletSpeed);

    // ================= 状态参数 =================
    std::vector<cv::Point2f> m_camera_points;
    std::vector<cv::Point3f> m_world_points;
    auto_buff::Direction m_direction; // 旋转方向
    auto_buff::Convexity m_convexity; // 拟合数据凹凸性
    int m_total_shift;                // 总体的装甲板切换数
    double m_bullet_speed;            // 子弹速度
    bool m_first_detect;              // 第一次检测的标志位，第一次检测有效之后置为 true

    std::vector<double> m_direction_data; // 计算旋转方向的数据

    std::vector<std::pair<double, double>> m_fit_data; // 拟合数据
    double m_angleRel;              // 这一帧相对于第一帧的旋转角度（去除了装甲板切换的影响）
    std::array<double, 5> m_params; // 拟合参数
    std::chrono::steady_clock::time_point m_frame_time; // 当前帧的时间戳
    std::chrono::steady_clock::time_point m_start_time; // 开始的时间戳

    double m_receive_roll;  // 当前帧的roll
    double m_receive_pitch; // 当前帧的pitch
    double m_receive_yaw;   // 当前帧的yaw
    double m_predict_pitch;
    double m_predict_yaw;

    cv::Point3f m_predict_robot; // 预测击打的机器人坐标
    cv::Point2f m_predict_pixel; // 预测击打的像素坐标

    cv::Mat m_matW2C;      // 世界坐标系转相机坐标系的 4x4 变换矩阵
    cv::Mat m_matC2G;      // 相机坐标系转云台坐标系的 4x4 变换矩阵
    cv::Mat m_matG2R;      // 云台坐标系转机器人坐标系的 4x4 变换矩阵
    cv::Mat m_matW2R;      // 世界坐标系转机器人坐标系的 4x4 变换矩阵
    cv::Mat m_rMatW2R;     // 世界坐标系转机器人坐标系的 3x3 旋转矩阵
    cv::Mat m_rMatW2RBase; // 第一次检测有效的世界坐标系转机器人坐标系的 3x3 旋转矩阵
    double m_angle_rel;    // 这一帧相对于第一帧的旋转角度（去除了装甲板切换的影响）
    double m_angle_last;   // 上一帧相对于第一帧的旋转角度（不考虑装甲板切换
    double m_distance2target;

    cv::Point3f m_armor_robot;  // 装甲板中心的机器人坐标
    cv::Point3f m_center_robot; // 中心 R 的机器人坐标

    int m_direction_thresh;
    std::thread m_fit_thread;
    std::shared_mutex m_mutex;

    // ================= 初始化参数 =================
    inline static int FPS;
    inline static int min_fit_data_size;
    inline static int max_fit_data_size;
    inline static double current_bullet_speed;
    inline static double min_bullet_speed;
    inline static double default_bullet_speed;

    inline static cv::Mat intrinsic_matrix;
    inline static cv::Mat dist_coeffs;

    inline static std::array<double, 3> camera2gimbal_translation;
    inline static const std::array<double, 3> camera2gimbal_rotation{0.0, 0.0, 0.0};

    inline static const double min_distance_to_target{4};
    inline static const double max_distance_to_target{10};

    inline static const double angle_between_fan_blades{72 * CV_PI / 180};

    inline static double compansate_time;
    inline static double compansate_pitch;
    inline static double compansate_yaw;

    inline static const double small_power_rune_rotation_speed{1.04719};

    // 重力
    inline static const double gravity{9.800};
  };

  Convexity get_convexity(const std::vector<std::pair<double, double>>& data);
  std::array<double, 5> ransac_fitting(const std::vector<std::pair<double, double>>& data,
                                       Convexity convexity);
  std::array<double, 5> least_square_estimate(const std::vector<std::pair<double, double>>& points,
                                              const std::array<double, 5>& params,
                                              Convexity convexity);
  cv::Mat world2Camera(const std::vector<cv::Point3f>& worldPoints,
                       const std::vector<cv::Point2f>& cameraPoints, const cv::Mat& intrinsicMatrix,
                       const cv::Mat& distCoeffs);
  cv::Mat camera2Gimbal(const std::array<double, 3>& r, const std::array<double, 3>& t);
  cv::Mat gimbal2Robot(double pitch, double yaw, double roll);

  std::pair<double, double> getPitchYawFromRobotCoor(const cv::Point3f& target, double bulletSpeed);

  /**
   * @brief 惩罚项，让拟合的参数更加贴近预设的参数
   */
  class CostFunctor1 : public ceres::SizedCostFunction<1, 5>
  {
  public:
    CostFunctor1(double truth_, int id_)
        : truth(truth_)
        , id(id_)
    {
    }
    virtual ~CostFunctor1() {};
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const
    {
      double pre = parameters[0][id];
      residuals[0] = pre - truth;
      if (jacobians != nullptr) {
        if (jacobians[0] != nullptr) {
          for (int i = 0; i < 5; ++i) {
            if (i == id) {
              jacobians[0][i] = 1;
            } else {
              jacobians[0][i] = 0;
            }
          }
        }
      }
      return true;
    }
    double truth;
    int id;
  };

  /**
   * @brief 拟合项
   */
  class CostFunctor2 : public ceres::SizedCostFunction<1, 5>
  {
  public:
    CostFunctor2(double t_, double y_)
        : t(t_)
        , y(y_)
    {
    }
    virtual ~CostFunctor2() {};
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const
    {
      double a = parameters[0][0];
      double w = parameters[0][1];
      double t0 = parameters[0][2];
      double b = parameters[0][3];
      double c = parameters[0][4];
      double cs = cos(w * (t + t0));
      double sn = sin(w * (t + t0));
      residuals[0] = -a * cs + b * t + c - y;
      if (jacobians != NULL) {
        if (jacobians[0] != NULL) {
          jacobians[0][0] = -cs;
          jacobians[0][1] = a * (t + t0) * sn;
          jacobians[0][2] = a * w * sn;
          jacobians[0][3] = t;
          jacobians[0][4] = 1;
        }
      }
      return true;
    }
    double t, y;
  };

} // namespace auto_buff
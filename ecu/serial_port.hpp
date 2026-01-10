#pragma once

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "ecu/command.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

// // 哨兵专有
// enum ShootMode { left_shoot, right_shoot, both_shoot };
// const std::vector<std::string> SHOOT_MODES = {"left_shoot", "right_shoot", "both_shoot"};

namespace ecu
{
  enum Mode { idle, auto_aim, small_buff, big_buff, outpost };
  const std::vector<std::string> MODES = {"idle", "auto_aim", "small_buff", "big_buff", "outpost"};

  struct IMUData {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
  };

  class SerialPort
  {
  public:
    double bullet_speed;
    Mode mode;
    // ShootMode shoot_mode; 烧饼
    double ft_angle; // flight time angle

    explicit SerialPort(const std::string& config_path);
    ~SerialPort();

    Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);

    void send(Command command) const;

  private:
    tools::ThreadSafeQueue<IMUData> queue_;

    int fd_; // 串口文件描述符
    std::thread read_thread_;
    std::atomic<bool> running_;

    std::string port_;
    int baudrate_;

    IMUData data_ahead_;
    IMUData data_behind_;

    void readLoop();
    void parseFrame(const std::vector<uint8_t>& frame);
    std::string read_yaml(const std::string& config_path);
  };

} // namespace ecu

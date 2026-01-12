#pragma once

#include "rune_detector.hpp"
#include "rune_solver.hpp"
#include <string>

namespace auto_buff
{
  class PowerRune
  {
  public:
    PowerRune(const std::string& config_path)
        : buff_detection(config_path)
        , buff_solver(config_path)
    {
      tools::logger()->debug("PowerRune initialized with config: {}", config_path);
    };
    bool run_once(const cv::Mat& img, double pitch, double yaw, double roll = 0.0);
    void run_once_debug(const cv::Mat& image, double pitch, double yaw, double roll = 0.0);

  private:
    auto_buff::BuffDetection buff_detection;
    auto_buff::BuffSolver buff_solver;
  };
} // namespace auto_buff
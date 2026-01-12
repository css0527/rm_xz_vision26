#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include <fmt/chrono.h>
#include "armor.hpp"

namespace xz_vision
{
  class Classifier
  {
  public:
    explicit Classifier(const std::string& config_path);

    void classify(Armor& armor);

    void ovclassify(Armor& armor);

  private:
    cv::dnn::Net net_;
    ov::Core core_;
    ov::CompiledModel compiled_model_;
  };

} // namespace auto_aim

#include <fmt/core.h>

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "function/auto_aim/detector.hpp"
#include "tools/exiter.hpp"
#include "tools/draw_tool.hpp"
#include "tools/logger.hpp"
#include "tools/math_tool.hpp"
#include "tools/plotter.hpp"
#include "ecu/camera.hpp"
#include "ecu/command.hpp"

using namespace cv;
using namespace std;

const std::string keys = "{help h usage ? |     | 输出命令行参数说明 }"
                         "{@config-path c | ../configs/how_to_set_params.yaml | "
                         "yaml配置文件的路径}";

int main(int argc, char* argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  // VideoCapture cap("/home/chaichai/project/rm_xz_vision26/assets/example.mp4");
  ecu::Camera Camera(config_path); // 添加的

  //   if (!cap.isOpened()) {
  //     std::cerr << "无法打开视频文件!" << std::endl;
  //     return -1;
  //   }

  xz_vision::Detector detector(config_path, true);

  while (true) {
    cv::Mat raw_img;
    // cap >> raw_img;

    // if (raw_img.empty())
    //   break;

    std::chrono::steady_clock::time_point timestamp;

    Camera.read(raw_img, timestamp);

    auto t_now = std::chrono::steady_clock::now();

    auto armors = detector.detect(raw_img); // 直接调用 detect 方法

    auto key = cv::waitKey(30);
    if (key == 'q')
      break;
  }
  return 0;
}
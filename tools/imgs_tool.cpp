#include "imgs_tool.hpp"
#include <opencv2/opencv.hpp>

namespace tools
{
  /**
   * @brief 根据图像的大小调整 roi 位置，使其不越界导致程序终止
   * @param[in] rect          待调整的 roi
   * @param[in] rows          行数
   * @param[in] cols          列数
   */
  void reset_roi(cv::Rect2f& rect, int rows, int cols)
  {
    // 调整左上角点的坐标
    rect.x = rect.x < 0 ? 0 : rect.x >= cols ? cols - 1 : rect.x;
    rect.y = rect.y < 0 ? 0 : rect.y >= rows ? rows - 1 : rect.y;
    // 调整长宽
    rect.width = rect.x + rect.width >= cols ? cols - rect.x - 1 : rect.width;
    rect.height = rect.y + rect.height >= rows ? rows - rect.y - 1 : rect.height;
    // 此时可能出现 width 或 height 小于 0 的情况，因此需要将其置为 0
    if (rect.width < 0) {
      rect.width = 0;
    }
    if (rect.height < 0) {
      rect.height = 0;
    }
  }

  /**
   * @brief 根据图像的大小调整 roi 位置，使其不越界导致程序终止
   * @param[in] rect          待调整的 roi
   * @param[in] mat           图像
   */
  void reset_roi(cv::Rect2f& rect, const cv::Mat& mat) { reset_roi(rect, mat.rows, mat.cols); }

  void reset_roi(cv::Rect2f& rect, const cv::Rect2f& lastRoi)
  {
    reset_roi(rect, static_cast<int>(lastRoi.height), static_cast<int>(lastRoi.width));
  }

  /**
   * @brief 判断点是否在矩形内部（包括边界）
   * @param[in] point
   * @param[in] rect
   * @return true
   * @return false
   */
  bool in_rect(const cv::Point2f& point, const cv::Rect2f& rect)
  {
    return point.x >= rect.x && point.x <= rect.x + rect.width && point.y >= rect.y &&
           point.y <= rect.y + rect.height;
  }

} // namespace tools

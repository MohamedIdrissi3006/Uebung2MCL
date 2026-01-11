#include "mcl_localization/mcl.hpp"

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

class MCLNode : public rclcpp::Node
{
public:
  MCLNode()
  : Node("mcl_node"), mcl_(200)
  {
    mcl_.initializeUniform(-10.0, 10.0, -10.0, 10.0);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/robot_noisy", 10,
      std::bind(&MCLNode::odomCallback, this, std::placeholders::_1));

    landmarks_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/landmarks_observed", 10,
      std::bind(&MCLNode::landmarksCallback, this, std::placeholders::_1));

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "/mcl_pose", 10);
  }

private:
  MCL mcl_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr landmarks_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

  double last_x_ = 0.0;
  double last_y_ = 0.0;
  double last_theta_ = 0.0;
  bool first_odom_ = true;

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;

    double siny = 2.0 * msg->pose.pose.orientation.z *
                  msg->pose.pose.orientation.w;
    double cosy = 1.0 - 2.0 * msg->pose.pose.orientation.z *
                          msg->pose.pose.orientation.z;
    double theta = std::atan2(siny, cosy);

    if (!first_odom_)
      mcl_.motionUpdate(x - last_x_, y - last_y_, theta - last_theta_);

    first_odom_ = false;
    last_x_ = x;
    last_y_ = y;
    last_theta_ = theta;
  }

  void landmarksCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    std::vector<double> obs_x, obs_y;
    std::vector<double> lm_x, lm_y;

    // (Assume known landmark order for simplicity)
    // Fill vectors here from PointCloud2

    mcl_.measurementUpdate(obs_x, obs_y, lm_x, lm_y, 0.2);
    mcl_.resample();

    auto est = mcl_.estimatePose();

    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = now();
    pose.header.frame_id = "map";
    pose.pose.position.x = est.x;
    pose.pose.position.y = est.y;
    pose.pose.orientation.z = std::sin(est.theta / 2.0);
    pose.pose.orientation.w = std::cos(est.theta / 2.0);

    pose_pub_->publish(pose);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MCLNode>());
  rclcpp::shutdown();
  return 0;
}

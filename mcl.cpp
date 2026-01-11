#include "/home/studamr/ros2_mcl_ws/src/mcl_localization/include/mcl_localization/mcl.hpp"
#include <cmath>
#include <numeric>

MCL::MCL(int num_particles)
: num_particles_(num_particles)
{
  rng_.seed(std::random_device{}());
  particles_.resize(num_particles_);
}

void MCL::initializeUniform(double xmin, double xmax,
                            double ymin, double ymax)
{
  std::uniform_real_distribution<double> dx(xmin, xmax);
  std::uniform_real_distribution<double> dy(ymin, ymax);
  std::uniform_real_distribution<double> dtheta(-M_PI, M_PI);

  for (auto &p : particles_)
  {
    p.x = dx(rng_);
    p.y = dy(rng_);
    p.theta = dtheta(rng_);
    p.weight = 1.0 / num_particles_;
  }
}

void MCL::motionUpdate(double dx, double dy, double dtheta)
{
  for (auto &p : particles_)
  {
    p.x += dx + noise_(rng_) * 0.02;
    p.y += dy + noise_(rng_) * 0.02;
    p.theta = normalizeAngle(p.theta + dtheta + noise_(rng_) * 0.01);
  }
}

void MCL::measurementUpdate(const std::vector<double>& obs_x,
                            const std::vector<double>& obs_y,
                            const std::vector<double>& lm_x,
                            const std::vector<double>& lm_y,
                            double sigma)
{
  double sigma2 = sigma * sigma;

  for (auto &p : particles_)
  {
    double weight = 1.0;

    for (size_t i = 0; i < obs_x.size(); ++i)
    {
      double dx = lm_x[i] - p.x;
      double dy = lm_y[i] - p.y;

      double c = std::cos(p.theta);
      double s = std::sin(p.theta);

      double pred_x =  c * dx + s * dy;
      double pred_y = -s * dx + c * dy;

      double ex = obs_x[i] - pred_x;
      double ey = obs_y[i] - pred_y;

      weight *= std::exp(-(ex*ex + ey*ey) / (2.0 * sigma2));
    }

    p.weight = weight;
  }

  // normalize
  double sum_w = 0.0;
  for (const auto &p : particles_) sum_w += p.weight;
  for (auto &p : particles_) p.weight /= sum_w;
}

void MCL::resample()
{
  std::vector<Particle> new_particles;
  new_particles.resize(num_particles_);

  double step = 1.0 / num_particles_;
  std::uniform_real_distribution<double> dist(0.0, step);
  double r = dist(rng_);

  double c = particles_[0].weight;
  int i = 0;

  for (int m = 0; m < num_particles_; ++m)
  {
    double U = r + m * step;
    while (U > c)
    {
      i++;
      c += particles_[i].weight;
    }
    new_particles[m] = particles_[i];
    new_particles[m].weight = step;
  }

  particles_ = new_particles;
}

MCL::Particle MCL::estimatePose() const
{
  Particle est{0, 0, 0, 1.0};

  for (const auto &p : particles_)
  {
    est.x += p.x * p.weight;
    est.y += p.y * p.weight;
    est.theta += p.theta * p.weight;
  }

  est.theta = normalizeAngle(est.theta);
  return est;
}

const std::vector<MCL::Particle>& MCL::particles() const
{
  return particles_;
}

double MCL::normalizeAngle(double angle) const
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

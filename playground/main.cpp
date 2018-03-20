#include <iostream>
#include <math.h>
#include "Eigen/Dense"
#include "../NumericalIntegration.h"

using namespace kalmanfilter;

struct Dynamics {
    using Scalar = float;
    using State = Eigen::Matrix<Scalar, 3, 1>; // x,y,theta
    using Control = Eigen::Matrix<Scalar, 3, 1>;
    using Jacobian = Eigen::Matrix<Scalar, 3, 3>;

    const Scalar B =  0.1; // [m] distance between front and back wheel

    State operator()(const State &x, const Control &u)
    {
        const Scalar v = u[0];
        const Scalar phi = u[1];
        const Scalar theta = x[2];
        State dx(
            v * cosf(theta),
            v * sinf(theta),
            v / B * tanf(phi)
            );
        return dx;
    }

    Jacobian jacobian(const Eigen::Vector3f &x, const Control &u)
    {
        const Scalar v = u[0];
        const Scalar theta = x[2];
        Jacobian J;
        J.setZero();
        J.coeffRef(0, 2) = -v * sinf(theta);
        J.coeffRef(1, 2) = v * cosf(theta);
        return J;
    }
};

struct Observation {
    using Scalar = float;
    using Measurement = Eigen::Matrix<Scalar, 3, 1>;
    using Jacobian = Eigen::Matrix<Scalar, 3, 3>;

    Measurement operator()(const Eigen::Vector3f &x)
    {
        return x;
    }

    Jacobian jacobian(const Eigen::Vector3f &x)
    {
        Eigen::Matrix3f H;
        H.setIdentity();
        return H;
    }

    Measurement from_sensor(const Eigen::Vector3f &x,
                            float acc_x,
                            float acc_y,
                            float gyro_z,
                            float delta_t)
    {
        Scalar theta, dtheta, dx;
        theta = x[2];
        dtheta = gyro_z * delta_t;
        dx = (0.5f * acc_x * delta_t + acc_y / gyro_z) * delta_t;
        Measurement y(
            x[0] + dx * cosf(theta),
            x[1] + dx * sinf(theta),
            x[2] + dtheta
            );
        return y;
    }
};

using Scalar = float;

int main(int argc, char *argv[])
{
    NumericalIntegration<Dynamics> f;
}

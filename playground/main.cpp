#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <math.h>
#include "Eigen/Dense"
#include "../NumericalIntegration.h"

using namespace kalmanfilter;

struct Dynamics {
public:
    static constexpr unsigned nx = 4;
    static constexpr unsigned nu = 2;
    using Scalar = float;
    using State = Eigen::Matrix<Scalar, nx, 1>; // x,y,v,theta
    using Control = Eigen::Matrix<Scalar, nu, 1>;
    using Jacobian = Eigen::Matrix<Scalar, nx, nx>;
    using ProcessNoiseCov = Eigen::Matrix<Scalar, nx, nx>;

    ProcessNoiseCov Q;
    const Scalar B =  0.1; // [m] base distance between front and back wheel

    Dynamics()
    {
        Q.setZero();
        Q.diagonal() << 0.01f, 0.01f, 0.01f, 0.01f;
    }

    State operator()(const State &x, const Control &u)
    {
        const Scalar v = u[0];
        const Scalar phi = u[1];
        const Scalar theta = x[3];
        State dx(
            v * cosf(theta),
            v * sinf(theta),
            0,
            v / B * tanf(phi)
            );
        return dx;
    }

    Jacobian jacobian(const State &x, const Control &u)
    {
        const Scalar v = u[0];
        const Scalar phi = u[1];
        const Scalar theta = x[3];
        const Scalar ct = cosf(theta);
        const Scalar st = sinf(theta);
        Jacobian J;
        J.setZero();
        J.coeffRef(0, 2) = ct;
        J.coeffRef(0, 3) = -v * st;
        J.coeffRef(1, 2) = st;
        J.coeffRef(1, 3) = v * ct;
        J.coeffRef(3, 2) = 1 / B * tanf(phi);
        return J;
    }
};

struct Observation {
    using Scalar = float;
    static constexpr unsigned nx = 4;
    static constexpr unsigned nz = 4;
    using State = Eigen::Matrix<Scalar, nx, 1>;
    using Measurement = Eigen::Matrix<Scalar, nz, 1>;
    using MeasurementNoiseCov = Eigen::Matrix<Scalar, nz, nz>;
    using Jacobian = Eigen::Matrix<Scalar, nz, nz>;

    MeasurementNoiseCov R;

    Observation()
    {
        R.setZero();
        R.diagonal() << 2e-6f, 2e-6, 2e-4, 5e-5;
    }

    Measurement operator()(const State &x)
    {
        return x;
    }

    Jacobian jacobian(const State &x)
    {
        Jacobian H;
        H.setIdentity();
        return H;
    }
};

using State = Dynamics::State;
using Measurement = Observation::Measurement;

Measurement from_imu(State x, float acc_x, float acc_y, float gyro_z, float dt)
{
    Measurement z;
    const float v = x[2];
    const float theta = x[3];
    const float ct = cosf(theta);
    const float st = sinf(theta);
    z << (0.5f * dt * dt * acc_x + v) * ct + 0.5f * dt * dt * st * acc_y,
    (0.5f * dt * dt * acc_x + v) * st - 0.5f * dt * dt * ct * acc_y,
        dt * acc_x,
        dt * gyro_z;
    return z + x;
}

template <typename Dynamics, typename Observation>
class KalmanFilter {
public:
    using Scalar = typename Dynamics::Scalar;
    using State = typename Dynamics::State;
    using Control = typename Dynamics::Control;
    using Measurement = typename Observation::Measurement;
    using StateCov = Eigen::Matrix<Scalar, Dynamics::nx, Dynamics::nx>;
    static constexpr unsigned nx = Dynamics::nx;
    static constexpr unsigned nz = Observation::nz;

    NumericalIntegration<Dynamics> f;
    Observation h;
    State x; // state vector
    StateCov P; // state covariance

    KalmanFilter(State x0, StateCov P0): x(x0), P(P0)
    {

    }

    void predict(const Control &u, Scalar delta_t)
    {
        typename Dynamics::Jacobian A;
        A.setIdentity();

        A += delta_t*f.jacobian(x, u);
        x = f.integrate(delta_t, delta_t / 10, x, u);
        // x = A * x
        P = A * P * A.transpose() + f.Q;
    }

    void correct(const Measurement &z)
    {
        Measurement y;                      // innovation
        Eigen::Matrix<Scalar, Observation::nz, Observation::nz> S;    // innovation covariance
        Eigen::Matrix<Scalar, nx, nz> K;    // Kalman gain
        typename Observation::Jacobian H;   // jacobian of h
        Eigen::Matrix<Scalar, nx, nx> IKH;  // temporary matrix
        Eigen::Matrix<Scalar, nx, nx> I;
        I.setIdentity();

        H = h.jacobian(x);
        S = H * P * H.transpose() + h.R;
        K = P * H.transpose() * S.inverse();

        /* variant 2
           Eigen::Matrix<Scalar, nz, nz> Sinv;
           Eigen::Matrix<Scalar, nz, nz> Iz;
           Iz.setIdentity();
           Sinv = S.llt().solve(Iz);
           K = P * H.transpose() * Sinv;
         */

        /* variant 3
           Eigen::Matrix<Scalar, nz, nx> A;
           A = H*P.transpose();
           // using S = S.transpose()
           K = S.llt().solve(A).transpose();
         */

        y = z - h(x);
        IKH = (I - K * H);

        // measurement update
        x = x + K * y;
        P = IKH * P * IKH.transpose() + K * h.R * K.transpose();
    }

    State update(const Control &u, const Measurement &z, Scalar delta_t)
    {
        predict(u, delta_t);
        correct(z);
        return x;
    }
};

int main(int argc, char *argv[])
{

    Dynamics::Control u;
    Measurement z;
    State x;
    KalmanFilter<Dynamics, Observation>::StateCov P;

    x << 0, 0, 0, 0;
    P.setZero();
    P.diagonal() << 0.1f, 0.1f, 0.1f, 0.1f;

    KalmanFilter<Dynamics, Observation> ekf(x, P);

    const float Ts = 0.02;
    std::ifstream data("imu.txt");
    std::string line;
    while (std::getline(data, line)) {
        float v, phi;
        float ax, ay, gz;
        sscanf(&line[0], "%f,%f,%f,%f,%f", &v, &phi, &ax, &ay, &gz);
        u << v, phi;
        z = from_imu(x, ax, ay, gz, Ts);

        x = ekf.update(u, z, Ts);

        printf("%f,%f,%f,%f\n", x[0], x[1], x[2], x[3]);
    }
}

#ifndef NUMERICALINTEGRATION_H
#define NUMERICALINTEGRATION_H

namespace kalmanfilter {

enum IntegrationMode {
    EULER,
    RK4
};

/** Numerical integrator class template
 * \param Functor function to be integrated
 * \param IntegrationMode choose EULER or RK4 method
 */
template<typename Functor, IntegrationMode mode = EULER>
class NumericalIntegration : public Functor {
public:
    using State = typename Functor::State;
    using Control = typename Functor::Control;
    using Scalar = typename Functor::State::Scalar;
private:
    // Explicit Euler method
    inline State Euler(Scalar time_span,
                       unsigned nb_steps,
                       const State &x0,
                       const Control &u)
    {
        State x;
        const Scalar step = time_span / nb_steps;
        for (x = x0; nb_steps > 0; nb_steps--) {
            State dx = this->operator()(x, u);
            x += step * dx;
        }
        return x;
    }

    // 4th order Runge-Kutta (RK4) method
    inline State RungeKutta4(Scalar time_span,
                             unsigned nb_steps,
                             const State &x0,
                             const Control &u)
    {
        State x, k1, k2, k3, k4;
        const Scalar step = time_span / nb_steps;
        for (x = x0; nb_steps > 0; nb_steps--) {
            k1 = this->operator()(x, u);
            k2 = this->operator()(x + step / 2 * k1, u);
            k3 = this->operator()(x + step / 2 * k2, u);
            k4 = this->operator()(x + step  * k3, u);
            x += step / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
        return x;
    }

public:
    /** Integrate given functor
     * \param time_span integration time span
     * \param x0 initial value
     * \param u control input
     * \param nb_steps number of integration steps
     * \return integration result
     */
    State integrate(Scalar time_span,
                    const State &x0,
                    const Control &u,
                    unsigned nb_steps = 1)
    {
        switch (mode) {
            case EULER:
                return Euler(time_span, nb_steps, x0, u);

            case RK4: // fall through
            default:
                return RungeKutta4(time_span, nb_steps, x0, u);
        }
    }
};

} // end namespace kalmanfilter

#endif // NUMERICALINTEGRATION_H

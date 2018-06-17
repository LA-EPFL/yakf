import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(args):
    with open(args.sim) as file:
        sim = np.loadtxt(file, delimiter=',')

    with open(args.test) as file:
        data = np.loadtxt(file, delimiter=',')

    plt.plot(sim[:,0], sim[:,1], label='sim')
    plt.plot(data[:,0], data[:,1], label='KF')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory comparison')
    plt.legend(loc=0)
    plt.savefig('demo_comparison.pdf')
    plt.show()

    plt.plot(sim[:,2])
    plt.plot(data[:,2])
    plt.xlabel('t [s]')
    plt.ylabel('v [m/s]')
    plt.title('Velocity comparison')
    plt.savefig('demo_comparison_vel.pdf')
    plt.show()

    plt.plot(sim[:,3])
    plt.plot(data[:,3])
    plt.title('Heading comparison')
    plt.xlabel('t [s]')
    plt.ylabel('θ [rad]')
    plt.savefig('demo_comparison_theta.pdf')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sim')
    parser.add_argument('test')
    args = parser.parse_args()
    main(args)

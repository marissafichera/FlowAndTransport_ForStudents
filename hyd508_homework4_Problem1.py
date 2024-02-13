# Making 3D plots in python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

C_TO_K = 273.15
ATM_TO_PA = 101325
R = 287


def get_T_P_ranges(rho_ref_ratio, rho_la_ratio, range_variable, range_variable_name, reference_variable_name):
    print(f'Calculating range of {range_variable_name}')

    things = []

    ratios = [rho_ref_ratio, rho_la_ratio]
    labels = ['rho_igm vs. rho_ref', 'rho_igm vs. rho_la']

    for r, label in zip(ratios, labels):
        for ratio, thing in zip(r, range_variable):
            if (ratio <= 1.02) & (ratio >= 0.98):
                things.append(thing)
        print(f'{label} at reference {reference_variable_name}')
        print(f'min {range_variable_name} = {np.min(things)}')
        print(f'max {range_variable_name} = {np.max(things)}')


def problem_1a():
    # the cell below this one will make your 3D (X,Y,Z) plot - all you need to do is define
    # the appropriate variables here. T = temperature range; P = pressure range; R = constant
    T = np.linspace(0 + C_TO_K, 100 + C_TO_K, 1000)
    P = np.linspace(0.1 * ATM_TO_PA, 10 * ATM_TO_PA, 1000)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    X = T
    Y = P
    X, Y = np.meshgrid(X, Y)
    Z = (Y / (R * X))

    surf = ax.plot_surface(X, Y * 0.001, Z, cmap=cm.coolwarm)
    plt.xlabel('temperature (kelvin)')
    plt.ylabel('pressure (kPa)')
    fig.colorbar(surf, label='density(kg/m^3)')

    plt.show()

    return T, P, Z, ax, fig


def problem_1b():
    # part b: air density
    temp_ref = 25 + C_TO_K
    P_ref = 100000
    rho_ref = P_ref / (R * temp_ref)

    print(f'ref. density = {rho_ref} kg/m^3')

    # part b: thermal expansion coefficient
    alpha_not = 1 / temp_ref
    beta_not = 1 / P_ref

    print(f'alpha_not = {alpha_not} K^-1')
    print(f'beta_not = {beta_not} Pa^-1')

    return rho_ref, temp_ref, P_ref, alpha_not, beta_not


def problem_1c(T, P, temp_ref, P_ref, rho_ref, alpha_not, beta_not):
    # use this setup to help with part c. this is the bare minimum -
    # you'll need to calculate all the relevant variables from part b first, then the relevant density ratio.
    # use np.min and np.max to find min/max values for temps and pressures

    # first determine what temperatures at reference pressure:
    # calculate density for IGM at P_ref, then calculate the ratio of rho_igm to rho_ref
    rho_igm_Pref = P_ref / (R * T)
    rr1 = rho_igm_Pref / rho_ref

    # calculate density for linear approximation given by 4.34 and calc ratios to rho_igm_Pref
    rho_la = rho_ref - (rho_ref * alpha_not * (T - temp_ref)) + (rho_ref * beta_not * (P - P_ref))
    rr2 = rho_igm_Pref / rho_la

    get_T_P_ranges(rr1, rr2, range_variable=T, range_variable_name='temperature', reference_variable_name='pressure')

    # now determine what pressure range at reference temperature
    # calculate density for IGM at T_ref and calc ratio of rho_igm to rho_ref
    rho_igm_Tref = P / (R * temp_ref)
    rr3 = rho_igm_Tref / rho_ref

    # density for linear approximation given by 4.34 already calculated, just calc ratios to rho_igm_Tref
    rr4 = rho_igm_Tref / rho_la

    get_T_P_ranges(rr3, rr4, range_variable=P, range_variable_name='pressure', reference_variable_name='temperature')


def problem_1c_ambitious(T, P, Z, rho_ref, alpha_not, beta_not, temp_ref, P_ref, ax, fig):
    rr1 = Z / rho_ref

    rho_la = rho_ref - (rho_ref * alpha_not * (T - temp_ref)) + (rho_ref * beta_not * (P - P_ref))
    print(rho_la)
    rr2 = Z / rho_la
    print(rr2.shape)

    ratios = [rr1, rr2]

    temps, pressures = [], []
    igm_rhos = []

    k = 0
    m = 0
    for r in ratios:
        for i in np.arange(len(T)):
            for j in np.arange(len(P)):
                if (r[i, j] <= 1.02) & (r[i, j] >= 0.98):
                    temps.append(T[i, j])
                    pressures.append(P[i, j])
                    igm_rhos.append(Z[i, j])
                    k = i
                    m = j


        print(f'min, max temperatures: {np.min(temps)}, {np.max(temps)}')
        print(f'min, max pressures: {np.min(pressures)}, {np.max(pressures)}')
    print(k, m)
    A = np.array((temps, pressures, igm_rhos), dtype=float)
    temps = np.array(temps)
    pressures = np.array(pressures)
    igm_rhos = np.array(igm_rhos)
    print(igm_rhos.shape)
    # subT = np.reshape(temps, [k, m])
    # subP = np.reshape(pressures, [k, m])
    # subZ = np.reshape(igm_rhos, [k, m])

    # subX, subY = np.meshgrid(subT, subP)


def main():
    T, P, Z, ax, fig = problem_1a()
    rho_ref, temp_ref, P_ref, alpha_not, beta_not = problem_1b()
    problem_1c(T, P, temp_ref, P_ref, rho_ref, alpha_not, beta_not)

    # problem_1c_ambitious(T, P, Z, rho_ref, alpha_not, beta_not, temp_ref, P_ref, ax, fig)


if __name__ == '__main__':
    main()

"""For problem 4.3 part b: "Give a proper mathematical statement for the transient problem...", write your answer as a list:

- Model Domain - the domain of the mathematical model (range of variables in coordinate system)

- Coordinate system = cartesian, cylindrical, spherical, etc.

- Mathematical (differential) equation = appropriate mathematical model

- Initial conditions = condition at time=0 i.e. T(x,t=0)=

- Boundary conditions = conditions at boundaries, looks something like this: T(x=0,t=0)= ..... ; T(x=L,t=0)= .....,

- Properties and parameters = self-explanatory


"""

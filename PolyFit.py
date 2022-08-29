import numpy as np
import operator

def polyfit_3d(x, y, z, degree):
    fit_eq = []
    fit_eq1 = []

    # Using Polyfit
    fitxy = np.polyfit(x, y, degree)
    fitxz = np.polyfit(x, z, degree)

    x_val = np.linspace(max(x), 128, 100)
    y_extr = np.polyval(fitxy, x_val)
    z_extr = np.polyval(fitxz, x_val)

    x = np.append(x, x_val)
    #y = np.append(y, y_extr)
    #z = np.append(z, z_extr)

    if degree == 2:
        a = fitxy[0]
        b = fitxy[1]
        c = fitxy[2]

        a1 = fitxz[0]
        b1 = fitxz[1]
        c1 = fitxz[2]

        fit_equation = a * np.square(x) + b * x + c
        fit_equation1 = a1 * np.square(x) + b1 * x + c1

        fit_eq.append(fit_equation)
        fit_eq1.append(fit_equation1)

    elif degree == 1:
        a = fitxy[0]
        b = fitxy[1]

        a1 = fitxz[0]
        b1 = fitxz[1]

        fit_equation = a * x + b
        fit_equation1 = a1 * x + b1

        fit_eq.append(fit_equation)
        fit_eq1.append(fit_equation1)

    # IMPORTANT: plotting has to be in this order, plot points -> sort fit -> plot fit

    fit_eq = np.array(fit_eq).flatten()
    fit_eq1 = np.array(fit_eq1).flatten()

    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, fit_eq, fit_eq1), key=sort_axis)
    x, fit_eq, fit_eq1 = zip(*sorted_zip)
    # ax.plot(fit_eq, x, fit_eq1, 'r')  # fit line

    return fit_eq, fit_eq1, x
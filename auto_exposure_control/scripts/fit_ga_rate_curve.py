# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


med_irad = np.array([0.00006, 0.000083, 0.000110, 0.000240, 0.00266, 0.008])
ga_rate = np.array([15.0, 2.0, 1.5, 1.0, 0.008, 0.001])

kl = ga_rate[0] + np.log10(med_irad[0])
print("kl is {0}".format(kl))

kh = ga_rate[-1] / 10**(-med_irad[-1])
print("kh is {0}".format(kh))

#ga_poly_coefs= np.polyfit(med_irad[0:-1], ga_rate[0:-1], 2)
#
#ga_poly = np.poly1d(ga_poly_coefs)
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#ax.scatter(med_irad, ga_rate)
#
#x = np.arange(0.0, 0.01, 0.00002)
#y = ga_poly(x)
#plt.plot(x, y)



fig = plt.figure()
ax = fig.add_subplot(311)

xl = np.linspace(0, med_irad[0], 50)
yl= kl + (- np.log10(xl))
ax.plot(xl, yl)

ax = fig.add_subplot(312)
xh = np.linspace(med_irad[-1], 0.1, 50)
yh= kh * 10 ** (-xh)
ax.plot(xh, yh)

ax = fig.add_subplot(313)
xm = np.zeros((0, ))
ym = np.zeros((0, ))
for i in np.arange(0, med_irad.shape[0]-1, 1):
    x1 = med_irad[i]
    x2 = med_irad[i+1]
    y1 = ga_rate[i]
    y2 = ga_rate[i+1]

    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1

    xm_s = np.linspace(med_irad[i], med_irad[i+1], 50)
    ym_s = k * xm_s + b

    xm = np.hstack((xm, xm_s))
    ym = np.hstack((ym, ym_s))

ax.plot(xm, ym)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xl, yl)
ax.plot(xm, ym)
ax.plot(xh, yh)
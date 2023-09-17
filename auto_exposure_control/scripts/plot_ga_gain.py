# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def desiredGradientAscentRate(med_irrad):
    r1 = 0.000083
    r2 = 0.000240

    b = 2.5
    a = -6300.0

    if med_irrad < r1:
        return -np.log10(med_irrad) + (-2.080921907623926)
    elif med_irrad > r2:
        return 10.0 ** (-5000.763089798977 * (med_irrad - 0.000240))
    else:
        return a * med_irrad + b

def gainScaling(gain):
    return 1.0 - 0.28 * (gain - 1.0)

med_irrads = np.arange(0.0, 0.001000, 0.000010)

gains = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
labels = [str(g) for g in gains]
num_gains = gains.shape[0]
des_ga_rates = np.zeros((num_gains, med_irrads.shape[0]))

for i in range(num_gains):
    scale = gainScaling(gains[i])
    des_ga_rates[i, :] =  [scale * desiredGradientAscentRate(ir) for ir in med_irrads]

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(num_gains):
    ax.plot(med_irrads, des_ga_rates[i, :], label=labels[i])
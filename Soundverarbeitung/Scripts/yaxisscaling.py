import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

fig, ax = plt.subplots()

ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_ticks(np.arange(-2, 3, 0.25))

x = np.arange(-1, 1, 0.1)
plt.plot(x, x**2)
plt.show()
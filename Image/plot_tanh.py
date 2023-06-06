import matplotlib.pyplot as plt
import numpy as np
import math

def tanh(x):
    result = (math.e ** (x) - math.e **(-x)) / (math.e ** (x) + math.e ** (-x))
    return result

x = np.linspace(-10,10)
y = tanh(x)
y2 = y * 1.5


fig = plt.figure()
ax = fig.add_subplot(111)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.set_xticks([-10, -5, 0, 5, 10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_yticks([-1, -0.5, 0.5, 1])



plt.plot(x, y, label="tanh", color='red')
plt.plot(x, y2, label="1.5 * tanh", color='blue')

plt.legend()

plt.savefig("../Image/tanh.png")




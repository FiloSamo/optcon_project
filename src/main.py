
from Dynamics import dynamics
from Animation import animate
import matplotlib.pyplot as plt
import numpy as np  

TT = 10000
xx = np.zeros([4, TT])
xx[:,0] = np.array([0,0,0,0])

xx_d = np.zeros([4, TT])
xx_d[:,0] = np.array([0,1,0,0])

for tt in range(0, TT-1):
    xx_plus, A , B = dynamics(xx[:,tt], np.zeros([2,1]), 0.001)
    # xx_d[:,tt+1] = A@xx_d[:,tt]
    xx[:,tt+1] = xx_plus.squeeze()

plt.figure()
plt.plot(xx_d[0,:])
plt.plot(xx_d[1,:])
plt.plot(xx_d[2,:])
plt.plot(xx_d[3,:])
plt.legend(['q1', 'q2', 'q1_dot', 'q2_dot'])
plt.show()

animate(xx, 0.001)
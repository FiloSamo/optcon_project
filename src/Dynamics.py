import numpy as np
import Animation

def dynamics(xx, uu):
    xx = xx.squeeze()
    uu = uu.squeeze()

    m1 = 1
    m2 = 1
    l1 = 1
    l2 = 1
    r1 = 0.5
    r2 = 0.5
    I1 = 0.33
    I2 = 0.33
    g = 9.81
    f1 = 0.1
    f2 = 0.1
    dt = 0.001

    ns = 4
    ni = 1

    print(uu)
    M = np.array([[I1+I2+m1*r1**2+m2*(l1**2+r2**2)+2*m2*l1*r2*np.cos(xx[1]), I2+m2*r2**2+m2*l1*r2*np.cos(xx[1])],
                  [I2+m2*r2**2+m2*l1*r2*np.cos(xx[1]), I2+m2*r2**2]])
    
    C = np.array([[-m2*l1*r2*xx[3]*np.sin(xx[1])*(xx[3]+2*xx[2])],
                  [m2*l1*r2*np.sin(xx[1])*xx[2]**2]])
    
    G = np.array([[g*(m1*r1+m2*l1)*np.sin(xx[0])+g*m2*r2*np.sin(xx[0]+xx[1])],
                 [g*m2*r2*np.sin(xx[0]+xx[1])]])
    
    F = np.array([[f1, 0],
                  [0, f2]])
    
    xx = xx.reshape([4,1])
    uu = uu.reshape([2,1])

    xx_dot = np.zeros([ns,1])

    xx_dot[0:2] = xx[2:4]
    xx_dot[2:4] = np.linalg.inv(M)@(-C-F@xx[2:4]-G+uu)

    xx_plus = np.zeros([ns,1])

    xx_plus[0:2] = xx[0:2]+xx_dot[0:2]*dt
    xx_plus[2:4] = xx[2:4]+xx_dot[2:4]*dt
    
    print(M)
    print(C)
    print(G)
    print(F)
    print(xx_dot)
    print(xx_plus)

    return xx_plus

dynamics(np.zeros([4,1]),np.zeros([2,1]))

TT = 10000
xx = np.zeros([4, TT])
xx[:,0] = np.array([np.pi/2,0,0,0])

for tt in range(0, TT-1):
    xx_plus = dynamics(xx[:,tt], np.zeros([2,1]))
    xx[:,tt+1] = xx_plus.squeeze()

print(xx)

Animation.animate(xx, 0.001)
import numpy as np

ns = 4
ni = 1

def dynamics(xx, uu, dt):
    xx = xx.squeeze()

    m1 = 1
    m2 = 1
    l1 = 1
    l2 = 1
    r1 = 0.5
    r2 = 0.5
    I1 = 0.33
    I2 = 0.33
    g = 9.81
    f1 = 0.5
    f2 = 0.5

    M = np.array([[I1+I2+m1*r1**2+m2*(l1**2+r2**2)+2*m2*l1*r2*np.cos(xx[1]), I2+m2*r2**2+m2*l1*r2*np.cos(xx[1])],
                  [I2+m2*r2**2+m2*l1*r2*np.cos(xx[1]), I2+m2*r2**2]])
    
    C = np.array([[-m2*l1*r2*xx[3]*np.sin(xx[1])*(xx[3]+2*xx[2])],
                  [m2*l1*r2*np.sin(xx[1])*xx[2]**2]])
    
    G = np.array([[g*(m1*r1+m2*l1)*np.sin(xx[0])+g*m2*r2*np.sin(xx[0]+xx[1])],
                 [g*m2*r2*np.sin(xx[0]+xx[1])]])
    
    F = np.array([[f1, 0],
                  [0, f2]])
    
    # Continuous dynamics
    
    xx = xx.reshape([ns,1])
    uu = np.array([[uu[0]], [0]])

    xx_dot = np.zeros([ns,1])

    xx_dot[0:2] = xx[2:4]
    xx_dot[2:4] = np.linalg.inv(M)@(-C-F@xx[2:4]-G+uu)

    
    xx_plus = np.zeros([ns,1])

    xx_plus[0:2] = xx[0:2]+xx_dot[0:2]*dt
    xx_plus[2:4] = xx[2:4]+xx_dot[2:4]*dt


    xx = xx.squeeze()
    uu = uu.squeeze()
        
    # Derivative of M respect to q1
    dM_dq1 = np.array([[0, 0],[0, 0]])
    # Derivative of M respect to q2
    dM_dq2 = np.array([[-2*m2*l1*r2*np.sin(xx[1]), -m2*l1*r2*np.sin(xx[1])],
                       [-m2*l1*r2*np.sin(xx[1]), 0]])
    # Derivative of M respect to q
    dM_dq = np.array([dM_dq1, dM_dq2])

    # Derivative of C respect to q
    dC_dq = np.array([[0 , -m2*l1*r2*xx[3]*np.cos(xx[1]*(xx[3]+2*xx[2]))],
                      [0, -m2*l1*r2*(xx[2]**2)*np.cos(xx[1])]])

    # Derivative of C respect to q_dot
    dC_dq_dot = np.array([[-2*m2*l1*r2*np.sin(xx[1])*xx[3], -2*m2*l1*r2*np.sin(xx[1])*(xx[2]+xx[3])],
                          [2*m2*l1*r2*np.sin(xx[1])*xx[2], 0]])
    
    # Derivative of G respect to q
    dG_dq = np.array([[g*(m1*r1+m2*l1)*np.cos(xx[0])+g*m2*r2*np.cos(xx[0]+xx[1]), g*m2*r2*np.cos(xx[0]+xx[1])],
                      [g*m2*r2*np.cos(xx[0]+xx[1]), g*m2*r2*np.cos(xx[0]+xx[1])]])
    
    # Compute the matrix A of the linearized system
    A = np.zeros([ns,ns])
    A[0:2,2:4] = np.eye(2)

    A[2:4,0:2] = np.linalg.inv(M)@(-dC_dq -dG_dq -dM_dq@xx_dot[2:4].squeeze())
    A[2:4,2:4] = np.linalg.inv(M)@(-dC_dq_dot -F)

    # Compute the matrix B of the linearized system
    B = np.array([[0],[0],[np.linalg.inv(M)[0,0]],[np.linalg.inv(M)[1,0]]])

    A_dis = np.eye(ns) + A*dt
    B_dis = B*dt

    return xx_plus, A_dis, B_dis

#
# Dynamics 
# Matteo Bonucci, Filippo Samorì, Jacopo Subini
# 01/01/2025
#

import numpy as np
import sympy as sp

# Parameters
ns = 4 # number of states 
ni = 1 # number of inputs

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


def gravity(xx):
    '''
    Computes the gravity compensation term for the first joint of the double pendulum
    Args:
        - xx: state vector
    Returns:
        - gravity: gravity compensation term
    '''
    gravity = g*(m1*r1+m2*l1)*np.sin(xx[0])+g*m2*r2*np.sin(xx[0]+xx[1])
    return gravity


def dynamics(xx, uu, dt):
    '''
    Computes the dynamics of the double pendulum
    Args:
        - xx: state vector
        - uu: input vector
        - dt: discretization time
    Returns:
        - xx_plus: next state vector
    '''

    xx = xx.squeeze()

    # Matrices of the dynamics
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

    # Continuous dynamics
    xx_dot[0:2] = xx[2:4]
    xx_dot[2:4] = np.linalg.inv(M)@(-C -(F@xx[2:4]) -G + uu)

    xx_plus = np.zeros([ns,1]) # discretized state

    # Discrete dynamics
    xx_plus[0:2] = xx[0:2]+xx_dot[0:2]*dt
    xx_plus[2:4] = xx[2:4]+xx_dot[2:4]*dt

    return xx_plus.squeeze()

def linearize_dynamics_symbolic():
    '''
    Linearize the dynamics of the double pendulum

    Returns:
        - A_func: function that computes the Jacobian of x_dot w.r.t. x
        - B_func: function that computes the Jacobian of x_dot w.r.t. u
    '''
    
    # Define symbolic variables
    q1, q2, q1_dot, q2_dot, q1_dd, q2_dd,  u = sp.symbols('q1 q2 q1_dot q2_dot q1_dd q2_dd u')

    q = sp.Matrix([q1, q2]) # configuration variables
    q_dot = sp.Matrix([q1_dot, q2_dot]) #  velocity variables
    q_dd = sp.Matrix([q1_dd, q2_dd]) # acceleration variables
    x = sp.Matrix([q1, q2, q1_dot, q2_dot]) # state variables
    u = sp.Matrix([u]) # input variables

    # Define symbolic matrices
    
    M_sym = sp.Matrix([[I1+I2+m1*r1**2+m2*(l1**2+r2**2)+2*m2*l1*r2*sp.cos(q2), I2+m2*r2**2+m2*l1*r2*sp.cos(q2)],
                [I2+m2*r2**2+m2*l1*r2*sp.cos(q2), I2+m2*r2**2]])
    
    C_sym = sp.Matrix([[-m2*l1*r2*q2_dot*sp.sin(q2)*(q2_dot+2*q1_dot)],
                [m2*l1*r2*sp.sin(q2)*q1_dot**2]])
    
    G_sym = sp.Matrix([[g*(m1*r1+m2*l1)*sp.sin(q1)+g*m2*r2*sp.sin(q1+q2)],
                [g*m2*r2*sp.sin(q1+q2)]])
    
    F_sym = sp.Matrix([[f1, 0],
                [0, f2]])
    
    # Continuous dynamics
    x_dot_sym = sp.Matrix([q1_dot, q2_dot, 0, 0])
    x_dot_sym[2:4, :] = M_sym.inv() * (-C_sym - F_sym * sp.Matrix([q1_dot, q2_dot]) - G_sym + sp.Matrix([u[0], 0]))
    
    A_sym = x_dot_sym.jacobian(x) # Jacobian of x_dot w.r.t. x
    B_sym = x_dot_sym.jacobian(u) # Jacobian of x_dot w.r.t. u
    
    A_func = sp.lambdify((q1, q2, q1_dot, q2_dot, u), A_sym, 'numpy') # Convert A_sym to a function
    B_func = sp.lambdify((q1, q2, q1_dot, q2_dot, u), B_sym, 'numpy') # Convert B_sym to a function
    
    return A_func, B_func

def linearize_dynamics_numeric(xx, uu, A_func, B_func, dt):
    '''
    Linearize the dynamics of the double pendulum
    Args:
        - xx: state vector
        - uu: input vector
        - A_func: function that computes the Jacobian of x_dot w.r.t. x
        - B_func: function that computes the Jacobian of x_dot w.r.t. u
        - dt: discretization time
    Returns:
        - A_dis: discretized Jacobian of x_dot w.r.t. x
        - B_dis: discretized Jacobian of x_dot w.r.t. u
    '''
    # Define symbolic variables
    q1, q2, q1_dot, q2_dot = xx # state variables
    u = uu # input variables

    A = A_func(q1, q2, q1_dot, q2_dot, u) # Jacobian of x_dot w.r.t. x
    B = B_func(q1, q2, q1_dot, q2_dot, u) # Jacobian of x_dot w.r.t. u

    # Discretize the dynamics
    A_dis = np.eye(ns) + A*dt 
    B_dis = B*dt 
    
    return A_dis, B_dis
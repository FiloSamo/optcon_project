#
# Reference trajectory generation
# Matteo Bonucci, Filippo Samorì, Jacopo Subini
# 01/01/2025
#


import numpy as np
import Dynamics as dyn
import solver as sol
import matplotlib.pyplot as plt

# Parameters
m1 = dyn.m1
m2 = dyn.m2
l1 = dyn.l1
l2 = dyn.l2
r1 = dyn.r1
r2 = dyn.r2
I1 = dyn.I1
I2 = dyn.I2
g = dyn.g
f1 = dyn.f1
f2 = dyn.f2

ns = dyn.ns
ni = dyn.ni

def step(xx_init, xx_fin, tf, dt):
      '''
      Generates a step reference trajectory from xx_init to xx_fin in tf seconds

      Args:
            - xx_init: initial state
            - xx_fin: final state
            - tf: final time
            - dt: discretization time

      Returns:
            - xx_ref: reference state trajectory
            - uu_ref: reference input trajectory
      '''

      # Initialization of the reference points
      xx_i = np.zeros(dyn.ns)
      xx_f = np.zeros(dyn.ns)

      # Conversion to radians
      xx_i[0:2] = np.deg2rad(xx_init[0:2])
      xx_f[0:2] = np.deg2rad(xx_fin[0:2])

      TT = int(tf/dt) # number of time steps
      
      xx_ref = np.zeros((ns, TT)) # state sequence initialization
      uu_ref = np.zeros((ni, TT)) # input sequence initialization

      # Trajectory generation

      xx_ref[0,:int(TT/2)] = np.full(int(TT/2),xx_i[0])
      xx_ref[1,:int(TT/2)] = np.full(int(TT/2),xx_i[1])

      xx_ref[2,:int(TT/2)] = np.full(int(TT/2),xx_i[2])
      xx_ref[3,:int(TT/2)] = np.full(int(TT/2),xx_i[3])

      xx_ref[0,int(TT/2):] = np.full(int(TT/2),xx_f[0])
      xx_ref[1,int(TT/2):] = np.full(int(TT/2),xx_f[1])

      xx_ref[2,int(TT/2):] = np.full(int(TT/2),xx_f[2])
      xx_ref[3,int(TT/2):] = np.full(int(TT/2),xx_f[3])

      uu_ref[:,:] = dyn.gravity(xx_ref) # gravity compensation

      return xx_ref, uu_ref

def poly_3(xx_init, xx_fin, tf, dt):
      
      TT = int(tf/dt) # number of time steps

      # Initialization of the reference points
      xx_i = np.zeros(dyn.ns) 
      xx_f = np.zeros(dyn.ns)
      
      # Conversion to radians
      xx_i[0:2] = np.deg2rad(xx_init[0:2])
      xx_f[0:2] = np.deg2rad(xx_fin[0:2])

      # Initialization of the reference trajectory 
      xx_ref = np.zeros((ns, TT))
      uu_ref = np.zeros((ni, TT))

      # Trajectory generation for the first joint
      tt = np.linspace(0, tf/2 , int(TT/2))

      a0 = xx_i[0] 
      a1 = 0
      a2 = 3 * (xx_f[0] - xx_i[0]) / ((tf/2)**2)
      a3 = -2 * (xx_f[0] - xx_i[0]) / ((tf/2)**3)

      xx_ref[0,:int(TT/4)] = np.full(int(TT/4), xx_i[0])
      xx_ref[2,:int(TT/4)] = np.full(int(TT/4), xx_i[2])

      for i in range(0, int(TT/2)):
          xx_ref[0, int(TT/4) + i] = a0 + a1 * tt[i] + a2 * tt[i]**2 + a3 * tt[i]**3
      
      for i in range(0, int(TT/2)):
          xx_ref[2, int(TT/4) + i] = a1 + 2* a2 * tt[i] + 3* a3 * tt[i]**2
      
      xx_ref[0,int(3*TT/4):] = np.full(int(TT/4), xx_f[0])
      xx_ref[2,int(3*TT/4):] = np.full(int(TT/4), xx_f[2])

      # Trajectory generation for the second joint

      if xx_i[0] + xx_i[1]  == 0:
            xx_ref[1,:] = - xx_ref[0,:]
      else:
            xx_ref[1,:] = np.pi - xx_ref[0,:]

      xx_ref[3,:] = - xx_ref[2,:]

      uu_ref[:,:] = dyn.gravity(xx_ref) # gravity compensation
      return xx_ref, uu_ref

def smooth_comp(points, tf, dt):
      '''
      Generates a smooth reference trajectory from a list of points

      Args:
            - points: list of points
            - tf: final time
            - dt: discretization time

      Returns:
            - xx_ref: reference state trajectory
            - uu_ref: reference input trajectory
      '''
     
      TT = int(tf/dt)

      # Initialization of the reference trajectory 
      xx_ref = np.zeros((ns, TT))
      uu_ref = np.zeros((ni, TT))

      n = len(points)-1
     
      for i in range(n):
           xx_ref[:,int(i*TT/n):int((i+1)*TT/n)] , uu_ref[:,int(i*TT/n):int((i+1)*TT/n)] = poly_3(points[i], points[i+1], tf/n, dt) 
      
      return xx_ref, uu_ref

def step_comp(points, tf, dt):

      '''
      Generates a step reference trajectory from a list of points

      Args:
            - points: list of points
            - tf: final time
            - dt: discretization time

      Returns:
            - xx_ref: reference state trajectory
            - uu_ref: reference input trajectory
      '''
     
      TT = int(tf/dt)

      # Initialization of the reference trajectory 
      xx_ref = np.zeros((ns, TT))
      uu_ref = np.zeros((ni, TT))

      n = len(points)-1 
     
      for i in range(n):
            xx_ref[:,int(i*TT/n):int((i+1)*TT/n)] , uu_ref[:,int(i*TT/n):int((i+1)*TT/n)] = step(points[i], points[i+1], tf/n, dt) 
      
      return xx_ref, uu_ref

def init_guess(xx_ref, uu_ref, tf, dt):

      '''
      Generates an initial guess for the optimization problem

      Args:
            - xx_ref: reference state trajectory
            - uu_ref: reference input trajectory
            - tf: final time
            - dt: discretization time

      Returns:

            - xx_init: initial state trajectory guess
            - uu_init: initial input trajectory guess 
      '''

      TT = int(tf/dt) # number of time steps

      # Initialization of the initial guess
      QQt_r = np.diag([1000.0, 1000.0, 10, 10])
      RRt_r = 10*np.eye(dyn.ni) 

      uu_r = np.zeros((dyn.ni, TT)) # input sequence initialization
      xx_r = np.zeros((dyn.ns, TT)) # state sequence initialization

      A_f , B_f = dyn.linearize_dynamics_symbolic()
      xx_r[:,0] = xx_ref[:,0]
      
      KKt_r, sigma_r = sol.LQR_solver(xx_ref,uu_ref,dyn, A_f, B_f, QQt_r,RRt_r,tf,dt) # LQR gain initialization

      # Simulation of the system with LQR control
      for tt in range(TT-1):
            uu_r[:,tt] = KKt_r[:,:,tt] @ (xx_r[:,tt]-xx_ref[:,tt]) + uu_ref[:,tt] + sigma_r[:,tt] # LQR control
            xx_r[:,tt+1] = dyn.dynamics(xx_r[:, tt], uu_r[:, tt], dt)

      uu_r[:,-1] = uu_r[:,-2] # for plotting purposes
      return xx_r, uu_r
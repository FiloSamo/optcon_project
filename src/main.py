#
# Main
# Matteo Bonucci, Filippo Samorì, Jacopo Subini
# 01/01/2025
#

# import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# import robot dynamics
import Dynamics as dyn
# import solver
import solver as sol
# import cost functions
import cost_newton as cst
# import armijo stepsize selector
import armijo
# import reference trajectory generator
import reference_trajectory as ref_gen
# import animation
import Animation

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Ctrl-C to work despite plotting

plt.rcParams["figure.figsize"] = (10,8) # set default figure size
plt.rcParams.update({'font.size': 22}) # set default font size

#######################################
# Algorithm parameters
#######################################

# Newton parameters

max_iters = 1000 # maximum number of iterations for the Newton solver

fixed_stepsize = 1e-1 # fixed stepsize

term_cond = 1e-6  # termination condition

# ARMIJO PARAMETERS

stepsize_0 = 1

# Visualization parameters

visu_descent_plot = False # plot the descent curve of the cost function after Armijo
visu_animation = True # plot the animation of the robot
dinamic_plot = True # plot the trajectory of the robot during the optimization

#######################################
# Trajectory parameters
#######################################

dt = 0.01   # discretization time in seconds
ns = dyn.ns # number of states
ni = dyn.ni # number of inputs

# Symbolic linearization of the dynamics 
A_f , B_f = dyn.linearize_dynamics_symbolic()

######################################
# Reference curve and initial guess
###################################### 

# # Task 1 - step reference - RRt = 10
# points = [[0,0,0,0],[90,-90,0,0]] 
# tf = 10 # final time in seconds
# TT = int(tf/dt) # discrete-time samples
# xx_ref , uu_ref = ref_gen.step_comp(points, tf, dt) # Reference trajectory generation
# xx_init = np.repeat(xx_ref[:,0].reshape(-1,1), TT, axis=1)
# uu_init = np.repeat(uu_ref[:,0].reshape(-1,1), TT, axis=1)

# # Task 2 - smooth trajectory with 2 points - RRt = 10
# points = [[0,0,0,0],[90,-90,0,0]] 
# tf = 10 # final time in seconds
# TT = int(tf/dt) # discrete-time samples
# xx_ref , uu_ref = ref_gen.smooth_comp(points, tf, dt) # Reference trajectory generation
# xx_init , uu_init = ref_gen.init_guess(xx_ref,uu_ref,tf,dt) # Initialization with the LQR solution of the reference trajectory

# # Task 2 alternative - unstable smooth trajectory - RRt = 10
# points = [[0,180,0,0],[90,90,0,0]] 
# tf = 10 # final time in seconds
# TT = int(tf/dt) # discrete-time samples
# xx_ref , uu_ref = ref_gen.smooth_comp(points, tf, dt) # Reference trajectory generation
# xx_init , uu_init = ref_gen.init_guess(xx_ref,uu_ref,tf,dt) # Initialization with the LQR solution of the reference trajectory

# # Task 2 alternative - Complex smooth trajectory - RRt = 10
# points = [[-80,260,0,0],[80,100,0,0]] 
# tf = 10 # final time in seconds
# TT = int(tf/dt) # discrete-time samples
# xx_ref , uu_ref = ref_gen.smooth_comp(points, tf, dt) # Reference trajectory generation
# xx_init , uu_init = ref_gen.init_guess(xx_ref,uu_ref,tf,dt) # Initialization with the LQR solution of the reference trajectory

# # Task 2 - Controlled fall down - RRt = 10
# points = [[180,0,0,0],[0,180,0,0]]
# tf = 6 # final time in seconds
# TT = int(tf/dt) # discrete-time samples
# xx_ref , uu_ref = ref_gen.smooth_comp(points, tf, dt) # Reference trajectory generation
# xx_init , uu_init = ref_gen.init_guess(xx_ref,uu_ref,tf,dt) # Initialization with the LQR solution of the reference trajectory

# # Task 2 - Complex stable smooth trajectory - RRt = 10
# points = [[0,0,0,0],[80,-80,0,0],[-80,80,0,0],[135,-135,0,0],[-135,135,0,0]]
# tf = 40 # final time in seconds
# TT = int(tf/dt) # discrete-time samples
# xx_ref , uu_ref = ref_gen.smooth_comp(points, tf, dt) # Reference trajectory generation
# xx_init , uu_init = ref_gen.init_guess(xx_ref,uu_ref,tf,dt) # Initialization with the LQR solution of the reference trajectory

# Swing up -- RRt = 10
tf = 6 # final time in seconds
TT = int(tf/dt) # discrete-time samples
points = [[0,0,0,0],[180,0,0,0]]
xx_ref , uu_ref = ref_gen.smooth_comp(points, tf, dt) # Reference trajectory generation
xx_ref[1,:] = np.zeros(TT) # set the velocity to zero
xx_init = np.repeat(xx_ref[:,0].reshape(-1,1), TT, axis=1)
uu_init = np.repeat(uu_ref[:,0].reshape(-1,1), TT, axis=1)

# # Task 2 - round trajectory - RRt = 10
# points = [[0,180,0,0],[360,-180,0,0]] 
# tf = 10 # final time in seconds
# TT = int(tf/dt) # discrete-time samples
# xx_ref , uu_ref = ref_gen.smooth_comp(points, tf, dt) # Reference trajectory generation
# xx_init , uu_init = ref_gen.init_guess(xx_ref,uu_ref,tf,dt) # Initialization with the LQR solution of the reference trajectory

# Plot the reference trajectory
plt.figure("Reference")
plt.plot(xx_ref[0,:], label = 'x_ref[0]')
plt.plot(xx_ref[1,:] , label = 'x_ref[1]')
plt.legend()
plt.show()

plt.figure("Initial guess")
plt.plot(xx_init[0,:], label = 'x_init[0]')
plt.plot(xx_init[1,:] , label = 'x_init[1]')
plt.legend()
plt.show()

#######################################
# Optimization 
#######################################

xx_star, uu_star, JJ, descent, n_iter = sol.newton_solver(xx_ref, uu_ref, xx_init, uu_init, dyn, tf, dt, max_iters, dynamic_plot = True, fixed_stepsize=fixed_stepsize,term_cond=term_cond, armijio=True, visu_descent_plot=False, stepsize_0=stepsize_0)

############################
# Plots
############################

# cost and descent

plt.figure('cost')
plt.plot(np.arange(n_iter), JJ[:n_iter])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

plt.figure('Descent')
plt.plot(np.arange(n_iter), descent[:n_iter])
plt.xlabel('$k$')
plt.ylabel('$||delta U||^2$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

# optimal trajectory

tt_hor = np.linspace(0,tf, TT)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, xx_star[0,:], linewidth=2)
axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_0$')

axs[1].plot(tt_hor, xx_star[1,:], linewidth=2)
axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_1$')

axs[2].plot(tt_hor, xx_star[2,:], linewidth=2)
axs[2].plot(tt_hor, xx_ref[2,:], 'g--', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$x_2$')

axs[3].plot(tt_hor, xx_star[3,:], linewidth=2)
axs[3].plot(tt_hor, xx_ref[3,:], 'g--', linewidth=2)
axs[3].grid()
axs[3].set_ylabel('$x_3$')

axs[4].plot(tt_hor, uu_star[0,:],'r', linewidth=2)
axs[4].plot(tt_hor, uu_ref[0,:], 'r--', linewidth=2)
axs[4].grid()
axs[4].set_ylabel('$u$')
axs[4].set_xlabel('time')
  
plt.show()
 
# Animation of the optimal trajectory
if visu_animation:
  Animation.animate(xx_star,xx_star, dt)

#######################################
# TASK 3
# Linearization around the optimal trajectory and LQR control 
#######################################

# LQR parameters initialization

QQt_r = np.diag([1000.0, 1000.0, 10, 10])
RRt_r = 10*np.eye(ni)

# LQR control computation
KKt, simgat = sol.LQR_solver(xx_star, uu_star, dyn, A_f, B_f, QQt_r, RRt_r, tf, dt)

print("LQR gain computed")

uu_r = np.zeros((ni, TT)) # input sequence initialization
xx_r = np.zeros((ns, TT)) # state sequence initialization
delta = np.array([0.01, 0.01, 0, 0]) # perturbation

x0 = xx_star[:,0] # initial condition

xx_r[:,0] = x0 + delta # initial condition with perturbation

actuation_noise = np.random.normal(0, 0.01, (ni, TT)) # actuation noise
measure_noise = np.random.normal(0, 0.001, (ns, TT)) # measurement noise

# Simulation of the system with LQR control
for tt in range(TT-1):

    uu_r[:,tt] = KKt[:,:,tt] @ (xx_r[:,tt]-xx_star[:,tt] + measure_noise[:,tt]) + uu_star[:,tt] + actuation_noise[:,tt] + simgat[:,tt] # LQR control
    xx_r[:,tt+1] = dyn.dynamics(xx_r[:, tt], uu_r[:, tt], dt)  # state update

# Plot the results

uu_r[:,-1] = uu_r[:,-2] # fill the last input with the previous one
fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, xx_r[0,:], linewidth=2, label="x[0]")
axs[0].plot(tt_hor, xx_star[0,:], 'g--', linewidth=2, label="$x_{ref}[0]$")
axs[0].legend()
axs[0].set_ylabel("rad")
axs[0].grid()

axs[1].plot(tt_hor, xx_r[1,:], linewidth=2, label="x[1]")
axs[1].plot(tt_hor, xx_star[1,:], 'g--', linewidth=2, label="$x_{ref}[1]$")
axs[1].legend()
axs[1].set_ylabel("rad")
axs[1].grid()

axs[2].plot(tt_hor, xx_r[2,:], linewidth=2, label="x[2]")
axs[2].plot(tt_hor, xx_star[2,:], 'g--', linewidth=2, label="$x_{ref}[2]$")
axs[2].legend()
axs[2].set_ylabel("rad/s")
axs[2].grid()

axs[3].plot(tt_hor, xx_r[3,:], linewidth=2, label="x[3]")
axs[3].plot(tt_hor, xx_star[3,:], 'g--', linewidth=2 , label="$x_{ref}[3]$")
axs[3].legend()
axs[3].set_ylabel("rad/s")
axs[3].grid()

axs[4].plot(tt_hor, uu_r[0,:],'r', linewidth=2, label="u")
axs[4].plot(tt_hor, uu_star[0,:], 'r--', linewidth=2, label="$u_{ref}$")
axs[4].legend()
axs[4].set_ylabel("Nm")
axs[4].grid()
axs[4].set_xlabel("Time [s]")

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, xx_r[0,:] - xx_star[0,:], linewidth=2, label="error x0")
axs[0].grid()
axs[0].legend()

axs[1].plot(tt_hor, xx_r[1,:] - xx_star[1,:], linewidth=2, label="error x1")
axs[1].grid()
axs[1].legend()

axs[2].plot(tt_hor, xx_r[2,:] - xx_star[2,:], linewidth=2, label="error x2")
axs[2].grid()
axs[2].legend()

axs[3].plot(tt_hor, xx_r[3,:] - xx_star[3,:], linewidth=2, label="error x3")
axs[3].grid()
axs[3].legend()

axs[4].plot(tt_hor, uu_r[0,:] - uu_star[0,:],'r', linewidth=2, label="error u")
axs[4].grid()
axs[4].set_xlabel('time')
axs[4].legend()

plt.show()

# Animation of the LQR control
if visu_animation:
  Animation.animate(xx_r,xx_star, dt)

#######################################
# TASK 4
# Model predictive control
#######################################

# MPC parameters initialization

QQt_mpc = np.diag([10000.0, 10000.0, 10, 10])
RRt_mpc = 1*np.eye(ni)

t_p = 1# prediction horizon in seconds

TT_pred = int(t_p/dt) # prediction horizon in samples

# MPC control computation
AAt = np.zeros((ns,ns,TT))
BBt = np.zeros((ns,ni,TT))
uu_mpc = np.zeros((ni,TT))
xx_mpc = np.zeros((ns,TT))

actuation_noise = np.random.normal(0, 0.01, (ni, TT)) # actuation noise
measure_noise = np.random.normal(0, 0.001, (ns, TT)) # measurement noise
delta = np.array([0.01, 0.01, 0, 0]) # perturbation

xx_mpc[:,0] = x0 + delta # initial condition with perturbation


for tt in range(TT-TT_pred):
  AAt[:,:,tt], BBt[:,:,tt] = dyn.linearize_dynamics_numeric(xx_star[:,tt], uu_star[:,tt], A_f, B_f, dt) # linearization of the dynamics

# MPC control computation

for tt in range(0, TT-TT_pred-1):
  xx0_mpc = xx_mpc[:,tt] + measure_noise[:,tt] # initial condition for the MPC

  if tt%20 == 0:
    print("time {}".format(tt)) # print the current iteration sample

  # solve the MPC problem at the current time step
  uu_temp = sol.solver_linear_mpc(dyn, A_f, B_f, QQt_mpc, RRt_mpc, QQt_mpc, xx0_mpc, xx_star[:,tt:tt+TT_pred], uu_star[:,tt:tt+TT_pred],dt, T_pred=t_p)

  # simulate the system with the MPC control
  uu_mpc[:,tt] = uu_temp + uu_star[:,tt] + actuation_noise[:,tt]
  xx_mpc[:,tt+1] = dyn.dynamics(xx_mpc[:,tt], uu_mpc[:,tt], dt)

# Plot the results
uu_mpc[:,TT - TT_pred-1] = uu_mpc[:,TT - TT_pred -2] # fill the last input with the previous one

tt_hor = np.linspace(0,tf-TT_pred*dt,TT-TT_pred)
fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, xx_mpc[0,:TT-TT_pred], linewidth=2, label="x[0]")
axs[0].plot(tt_hor, xx_star[0,:TT-TT_pred], 'g--', linewidth=2, label="$x_{ref}[0]$")
axs[0].grid()
axs[0].legend()
axs[0].set_ylabel('rad')

axs[1].plot(tt_hor, xx_mpc[1,:TT-TT_pred], linewidth=2, label="x[1]")
axs[1].plot(tt_hor, xx_star[1,:TT-TT_pred], 'g--', linewidth=2, label="$x_{ref}[1]$")
axs[1].grid()
axs[1].legend()
axs[1].set_ylabel('rad')

axs[2].plot(tt_hor, xx_mpc[2,:TT-TT_pred], linewidth=2, label="x[2]")
axs[2].plot(tt_hor, xx_star[2,:TT-TT_pred], 'g--', linewidth=2, label="$x_{ref}[2]$")
axs[2].grid()
axs[2].legend()
axs[2].set_ylabel('rad/s')

axs[3].plot(tt_hor, xx_mpc[3,:TT-TT_pred], linewidth=2, label="x[3]")
axs[3].plot(tt_hor, xx_star[3,:TT-TT_pred], 'g--', linewidth=2, label="$x_{ref}[3]$")
axs[3].grid()
axs[3].legend()
axs[3].set_ylabel('rad/s')

axs[4].plot(tt_hor, uu_mpc[0,:TT-TT_pred],'r', linewidth=2, label="u")
axs[4].plot(tt_hor, uu_star[0,:TT-TT_pred], 'r--', linewidth=2, label="$u_{ref}$")
axs[4].grid()
axs[4].legend()
axs[4].set_ylabel('Nm')
axs[4].set_xlabel('time')

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, xx_mpc[0,:TT-TT_pred] - xx_star[0,:TT-TT_pred], linewidth=2, label="error x0")
axs[0].grid()
axs[0].legend()

axs[1].plot(tt_hor,xx_mpc[1,:TT-TT_pred] - xx_star[1,:TT-TT_pred], linewidth=2, label="error x1")
axs[1].grid()
axs[1].legend()

axs[2].plot(tt_hor, xx_mpc[2,:TT-TT_pred] - xx_star[2,:TT-TT_pred], linewidth=2, label="error x2")
axs[2].grid()
axs[2].legend()

axs[3].plot(tt_hor, xx_mpc[3,:TT-TT_pred] - xx_star[3,:TT-TT_pred], linewidth=2, label="error x3")
axs[3].grid()
axs[3].legend()

axs[4].plot(tt_hor, uu_mpc[0,:TT-TT_pred] - uu_star[0,:TT-TT_pred],'r', linewidth=2, label="error u")
axs[4].grid()
axs[4].set_xlabel('time')
axs[4].legend()

plt.show()

# Animation of the MPC control
if visu_animation:
  Animation.animate(xx_mpc[:,:TT-TT_pred], xx_star[:,:TT-TT_pred] , dt)


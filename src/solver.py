#
# solver 
# Matteo Bonucci, Filippo Samorì, Jacopo Subini
# 01/01/2025
#


import numpy as np
import cost_newton as cst
import matplotlib.pyplot as plt
import armijo

def newton_solver(xx_ref, uu_ref, xx_init, uu_init, dyn, tf, dt, max_iters, term_cond = 1e-6 , fixed_stepsize = 1e-1, armijio = False, dynamic_plot = False,visu_descent_plot = False, armijo_maxiters = 20, cc = 0.5, beta = 0.7, stepsize_0 = 1):
    
    """
    Newton-based solver for trajectory optimization.
    Parameters:

    - xx_ref (np.ndarray): Reference state trajectory.
    - uu_ref (np.ndarray): Reference input trajectory.
    - xx_init (np.ndarray): Initial state trajectory.
    - uu_init (np.ndarray): Initial input trajectory.
    - dyn (object): Dynamics model with methods for linearization and simulation.
    - tf (float): Final time.
    - dt (float): Time step.
    - max_iters (int): Maximum number of iterations.
    - term_cond (float, optional): Termination condition for descent direction. Default is 1e-1.
    - fixed_stepsize (float, optional): Fixed step size for updates. Default is 1e-1.
    - armijio (bool, optional): Flag to use Armijo step size selection. Default is False.
    - dynamic_plot (bool, optional): Flag to enable dynamic plotting. Default is False.
    - visu_descent_plot (bool, optional): Flag to enable visualization of descent plot. Default is False.
    - armijo_maxiters (int, optional): Maximum iterations for Armijo step size selection. Default is 20.
    - cc (float, optional): Armijo parameter for sufficient decrease condition. Default is 0.5.
    - beta (float, optional): Armijo parameter for step size reduction. Default is 0.7.
    - stepsize_0 (float, optional): Initial step size for Armijo. Default is 1.
    Returns:

    tuple: 
        - xx_star (np.ndarray): Optimized state trajectory.
        - uu_star (np.ndarray): Optimized input trajectory.
        - JJ (np.ndarray): Cost function values over iterations.
        - descent (np.ndarray): Descent direction values over iterations.
        - max_iters (int): Number of iterations performed.
    """


    ######################################
    # Arrays to store data
    ######################################
    x0 = xx_ref[:,0] # initial condition
    TT = int(tf/dt) # number of time steps
    ns = dyn.ns # number of states
    ni = dyn.ni # number of inputs

    xx = np.zeros((ns, TT, max_iters))   # state seq.
    uu = np.zeros((ni, TT, max_iters))   # input seq.

    PP = np.zeros((ns+1, ns+1, TT, max_iters)) # P matrix for Riccati Equation

    delta_u = np.zeros((ni, TT, max_iters))   # input correction
    delta_x = np.zeros((ns+1, TT, max_iters)) # Extended state correction

    delta_x[0,:,:] = np.ones((TT,max_iters)) # Initialisation of the first row of the delta_x as one

    lambd = np.zeros((ns,TT,max_iters)) # lambda vector for the costate equation

    dJ = np.zeros((ni, TT, max_iters)) # gradient of the cost function

    JJ = np.zeros(max_iters)      # collect cost
    descent = np.zeros(max_iters) # collect descent direction
    descent_arm = np.zeros(max_iters) # collect descent direction

    ######################################
    # Main
    ######################################

    print('-*-*-*-*-*-')

    # Initialize the iteration counter
    kk = 0

    # Initialize the state and input sequences
    xx[:,:,0] = xx_init
    uu[:,:,0] = uu_init

    A_f , B_f = dyn.linearize_dynamics_symbolic()

    if dynamic_plot:
        fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    # Loop over the iterations
    for kk in range(max_iters-1):

        ##################################
        # Cost calculation
        ##################################

        JJ[kk] = cst.cost_calculator(xx[:,:,kk], uu[:,:,kk], xx_ref[:,:], uu_ref[:,:], TT)

        ##################################
        # Descent direction calculation
        ##################################

        _, qT, QT = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1]) # terminal cost and temrinal gradient

        PP[:,:,TT-1,kk] = np.block([[0, qT.T],
                                    [qT, QT]]) # P matrix initialization
        
        lambd[:,TT-1,kk] = qT.squeeze() # costate variable initialization

        KKt = np.zeros((ni, ns+1, TT)) # LQR initialization

        for tt in reversed(range(TT-1)):  # integration backward in time

            _, qqt, rrt, QQt, RRt, SSt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])
            AAt, BBt = dyn.linearize_dynamics_numeric(xx[:,tt,kk], uu[:,tt,kk], A_f , B_f ,dt)

            cct = np.zeros((ns,1)) # cct initialization
            cc_temp = dyn.dynamics(xx[:,tt,kk],uu[:,tt,kk],dt) - xx[:,tt+1,kk]
            cct = cc_temp.reshape((ns,1))
            # Extended Q matrix
            QQt_e = np.block([[0, qqt.T],
                        [qqt, QQt]]) 
            # Extended S matrix
            SSt_e = np.block([rrt, SSt]) 

            # Extended R matrix
            RRt_e = RRt 

            # Extended A matrix
            AAt_e = np.block([[1, np.zeros((1,4))],
                        [cct, AAt]])
            
            # Extended B matrix
            BBt_e = np.block([[0],
                        [BBt]])
            
            ##################################
            # Riccati Equation - backward
            ##################################

            KKt[:,:,tt] = -np.linalg.inv(RRt_e + BBt_e.T @PP[:,:,tt+1,kk] @ BBt_e ) @ (SSt_e +BBt_e.T @ PP[:,:,tt+1,kk] @ AAt_e)
            PP[:,:,tt,kk] = QQt_e + AAt_e.T @ PP[:,:,tt+1,kk] @ AAt_e - KKt[:,:,tt].T @ (RRt_e + BBt_e.T @ PP[:,:,tt+1,kk] @ BBt_e ) @ KKt[:,:,tt]

            # Costate equation and gradient of the cost function
            lambd[:,tt,kk] = AAt.T @ lambd[:,tt+1,kk] + qqt.squeeze()
            dJ[:,tt,kk] = BBt.T @ lambd[:,tt+1,kk] + rrt.squeeze()
            
        ################################## 
        # Descent direction calculation with forward integration
        ##################################

        for tt in range(TT-1):
            AAt, BBt = dyn.linearize_dynamics_numeric(xx[:,tt,kk], uu[:,tt,kk], A_f , B_f ,dt)
            
            cct = np.zeros((ns,1)) # cct initialization
            cc_temp = dyn.dynamics(xx[:,tt,kk],uu[:,tt,kk],dt) - xx[:,tt+1,kk]
            cct = cc_temp.reshape((ns,1))

            # Extended A matrix
            AAt_e = np.block([[1, np.zeros((1,4))],
                        [cct, AAt]])
            
            # Extended B matrix
            BBt_e = np.block([[0],
                        [BBt]])
            
            delta_u[:,tt,kk] = KKt[:,:,tt] @ delta_x[:,tt,kk]
            delta_x[:,tt+1,kk] = AAt_e @ delta_x[:,tt,kk] + BBt_e @ delta_u[:,tt,kk]

            # Descent direction calculation
            descent[kk] += delta_u[:,tt,kk].T @ delta_u[:,tt,kk] # descent direction squared
            descent_arm[kk] += dJ[:,tt,kk].T @ delta_u[:,tt,kk]  # descent direction for Armijo

        ##################################
        # Stepsize selection - ARMIJO
        ##################################

        #copy the penultimate element of the input sequence to the last element
        delta_u[:,-1,kk] = delta_u[:,-2,kk] 

        if armijio:

            stepsize = armijo.select_stepsize(stepsize_0, armijo_maxiters, cc, beta,
                                        delta_u[:,:,kk], KKt , xx_ref, uu_ref, x0, xx[:,:,kk], 
                                        uu[:,:,kk], JJ[kk], descent_arm[kk], visu_descent_plot, dt)
        else:
            stepsize = fixed_stepsize

        # if the stepsize is zero, the algorithm stops   
        if stepsize == 0:
            max_iters = kk

        ############################
        # Update the current solution
        ############################

        # Temporary variables to store the updated state and input sequences
        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT))

        xx_temp[:,0] = x0 # initial condition

        # Update the state and input sequences using feedback integration
        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt,kk] + KKt[:,1:,tt] @(xx_temp[:,tt] - xx[:,tt,kk]) + stepsize * KKt[:,0,tt]
            xx_temp[:,tt+1]  = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt], dt)
            
        # copy the penultimate element of the input sequence to the last element
        uu_temp[:,-1] = uu_temp[:,-2] 

        # Update the state and input sequences
        xx[:,:,kk+1] = xx_temp.copy()
        uu[:,:,kk+1] = uu_temp.copy()

        #########################################################
        # PLOT at each iteration# 
        ########################################################

        if dynamic_plot:

            # Create a figure and subplots
            axs[0].cla()
            axs[1].cla()
            axs[2].cla()
            axs[3].cla()
            axs[4].cla()

            # Plot x[0] and x_ref[0] on the first subplot
            axs[0].plot(np.linspace(0, TT-1, TT), xx[0, :, kk], "b-", label="x[0]")
            axs[0].plot(np.linspace(0, TT-1, TT), xx_ref[0, :], "b--", label="$x_{ref}[0]$")
            axs[0].legend()
            axs[0].set_ylabel("rad")
            axs[0].grid()
            #axs[0].set_title("Comparison of x[0] and $x_{ref}[0] [---]$")

            # Plot x[1] and x_ref[1] on the second subplot
            axs[1].plot(np.linspace(0, TT-1, TT), xx[1, :, kk], "c-", label="x[1]")
            axs[1].plot(np.linspace(0, TT-1, TT), xx_ref[1, :], "c--", label="$x_{ref}[1]$")
            axs[1].legend()
            axs[1].set_ylabel("rad")
            axs[1].grid()
            #axs[1].set_title("Comparison of x[1] and $x_{ref}[1] [---]$")

            # Plot x[0] and x_ref[0] on the first subplot
            axs[2].plot(np.linspace(0, TT-1, TT), xx[2, :, kk], "b-", label="x[2]")
            axs[2].plot(np.linspace(0, TT-1, TT), xx_ref[2, :], "b--", label="$x_{ref}[2]$")
            axs[2].legend()
            axs[2].set_ylabel("rad/s")
            axs[2].grid()
            #axs[2].set_title("Comparison of x[0] and $x_{ref}[0] [---]$")

            # Plot x[1] and x_ref[1] on the second subplot
            axs[3].plot(np.linspace(0, TT-1, TT), xx[3, :, kk], "c-", label="x[3]")
            axs[3].plot(np.linspace(0, TT-1, TT), xx_ref[3, :], "c--", label="$x_{ref}[3]$")
            axs[3].legend()
            axs[3].set_ylabel("rad/s")
            axs[3].grid()
            #axs[3].set_title("Comparison of x[1] and $x_{ref}[1] [---]$")

            axs[4].plot(np.linspace(0, TT-1, TT), uu[0,:,kk], "r-", label="u" )
            axs[4].plot(np.linspace(0, TT-1, TT), uu_ref[0,:], "r--", label="$u_{ref}$" )
            axs[4].legend()
            axs[4].set_ylabel("Nm")
            axs[4].grid()
            axs[4].set_xlabel("Time [s]")
            #axs[4].set_title("Comparison of u and $u_{ref} [---]$")

            #plt.tight_layout()
            plt.pause(1e-4)



        ############################
        # Termination condition
        ############################

        print('Iter = {}\t descent = {:.3e}\t Cost = {:.3e}'.format(kk, descent[kk], JJ[kk]))

        if descent[kk] <= term_cond:
            # if the descent direction is below the threshold, the algorithm stops
            max_iters = kk
        
        if kk >= max_iters:
          break

    if kk == 0:
        max_iters = 1 

    xx_star = xx[:,:,max_iters]
    uu_star = uu[:,:,max_iters]
    uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

    return xx_star, uu_star, JJ[:max_iters+1], descent[:max_iters+1], max_iters+1

def LQR_solver(xx_star, uu_star, dyn, A_f, B_f, QQt, RRt, tf, dt):

    """
    LQR solver for the optimal control problem
    Args:
        - xx_star (np.ndarray): Optimal state trajectory.
        - uu_star (np.ndarray): Optimal input trajectory.
        - dyn (object): Dynamics model with methods for linearization and simulation.
        - A_f (np.ndarray): Symbolic linearized dynamics matrix.
        - B_f (np.ndarray): Symbolic linearized input matrix.
        - QQt (np.ndarray): State weights.
        - RRt (np.ndarray): Input weights.
        - tf (float): Final time.
        - dt (float): Time step.
    Returns:
        np.ndarray: LQR gain matrix
    """

    ni = dyn.ni
    ns = dyn.ns
    TT = int(tf/dt)

    # Regulator parameters initialization
    PP = np.zeros((4,4,TT))

    PP[:,:,TT-1] = QQt # P matrix initialization

    sigmat = np.zeros((ni,TT)) # sigma matrix initialization
    ppt = np.zeros((ns,TT)) # p matrix initialization

    KKt = np.zeros((ni, ns, TT)) # LQR gain initialization

    for tt in reversed(range(TT-1)):
        AAt, BBt = dyn.linearize_dynamics_numeric(xx_star[:,tt], uu_star[:,tt], A_f , B_f ,dt)
        cct = np.zeros((ns,1)) # cct initialization
        cc_temp = dyn.dynamics(xx_star[:,tt],uu_star[:,tt],dt) - xx_star[:,tt+1]

        cct = cc_temp.reshape((ns,1))

        KKt[:,:,tt] = -np.linalg.inv(RRt + BBt.T @PP[:,:,tt+1] @ BBt ) @ (BBt.T @ PP[:,:,tt+1] @ AAt) # LQR gain
        sigmat[:,tt] = -np.linalg.inv(RRt + BBt.T @ PP[:,:,tt+1] @ BBt ) @ (BBt.T @ ppt[:,tt+1] + BBt.T @ PP[:,:,tt+1]@ cct) # sigma matrix

        PP[:,:,tt] = QQt + AAt.T @ PP[:,:,tt+1] @ AAt - KKt[:,:,tt].T @ (RRt + BBt.T @ PP[:,:,tt+1] @ BBt ) @  KKt[:,:,tt]

        ppt[:,tt] = AAt.T @ ppt[:,tt+1] + AAt.T @ PP[:,:,tt+1] @ cct.squeeze() - KKt[:,:,tt].T @ (RRt + BBt.T @ PP[:,:,tt+1] @ BBt) @ sigmat[:,tt]
    
    
    return KKt , sigmat


def solver_linear_mpc(dyn, A_f, B_f, QQ, RR, QQf, xxt, xx_star, uu_star , dt, T_pred = 20):
    """
    Linear Unconstrained MPC solver for the optimal control problem
    Args:
        - dyn (object): Dynamics model with methods for linearization and simulation.
        - A_f (np.ndarray): Symbolic linearized dynamics matrix.
        - B_f (np.ndarray): Symbolic linearized input matrix.
        - QQt (np.ndarray): State weights.
        - RRt (np.ndarray): Input weights.
        - QQf (np.ndarray): Terminal state weights.
        - xxt (np.ndarray): Initial state trajectory.
        - xx_star (np.ndarray): Optimal state trajectory.
        - uu_star (np.ndarray): Optimal input trajectory.
        - dt (float): Time step.
        - T_pred (float): Prediction horizon in seconds (default is 20).
    Returns:
        - uu_mpc (np.ndarray): Optimal input at time t.
    """
    uu_mpc = 0# input seq.
    
    KKt, sigmat  = LQR_solver(xx_star, uu_star, dyn, A_f, B_f, QQ, RR, T_pred, dt) # LQR gain initialization

    uu_mpc = KKt[:,:,0] @ (xxt-xx_star[:,0]) + sigmat[:,0] # LQR control

    return uu_mpc

# def solver_linear_mpc(AA, BB, QQ, RR, QQf, xxt, xx_star, uu_star, umax = 100, umin = -100, x1_max = 2000, x1_min = -2000, x2_max = 2000, x2_min = -2000,  T_pred = 20):
#     """
#         Linear MPC solver - Constrained LQR

#         Given a measured state xxt measured at t
#         gives back the optimal input to be applied at t

#         Args
#           - AA, BB: linear dynamics
#           - QQ,RR,QQf: cost matrices
#           - xxt: initial condition (at time t)
#           - T: time (prediction) horizon

#         Returns
#           - u_t: input to be applied at t
#           - xx, uu predicted trajectory

#     """
#     xxt = xxt.squeeze()

#     ns, ni, _ = BB.shape

#     xx_mpc = cp.Variable((ns, T_pred))
#     uu_mpc = cp.Variable((ni, T_pred))

#     cost = 0
#     constr = []

#     for tt in range(T_pred-1):
#         cost += cp.quad_form(xx_mpc[:,tt]-xx_star[:,tt], QQ) + cp.quad_form(uu_mpc[:,tt]-uu_star[:,tt], RR)
#         constr += [xx_mpc[:,tt+1] == AA[:,:,tt]@xx_mpc[:,tt] + BB[:,:,tt]@uu_mpc[:,tt], # dynamics constraint
#                 uu_mpc[:,tt] <= umax, # other constraints
#                 uu_mpc[:,tt] >= umin,
#                 xx_mpc[0,tt] <= x1_max,
#                 xx_mpc[0,tt] >= x1_min,
#                 xx_mpc[1,tt] <= x2_max,
#                 xx_mpc[1,tt] >= x2_min]
#     # sums problem objectives and concatenates constraints.
#     cost += cp.quad_form(xx_mpc[:,T_pred-1], QQf)
#     constr += [xx_mpc[:,0] == xxt]

#     problem = cp.Problem(cp.Minimize(cost), constr)
#     problem.solve()

#     if problem.status == "infeasible":
#     # Otherwise, problem.value is inf or -inf, respectively.
#         print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

#     return uu_mpc[:,0].value
import numpy as np
import matplotlib.pyplot as plt
import cost_newton as cst
import Dynamics as dyn


def select_stepsize(stepsize_0, armijo_maxiters, cc, beta, deltau, KKt, xx_ref, uu_ref,  x0, xx, uu, JJ, descent_arm, plot = False, dt=0.01):

      """
      Computes the stepsize using Armijo's rule.
      input parameters:
            - stepsize_0 : initial stepsize guess,
            - armijo_maxiters : maximum number of iterations for armijo rule,
            - deltau : descending direction for the control action,
            - xx_ref : reference trajectory state,
            - uu_ref : reference trajectory input,
            - x0 : initial state,
            - uu : input at current iteration,
            - JJ : cost at current iteration,
            - descent_arm: armijo descent direction at current iteration,
            - plot: whether or not to show descent plot.

      output parameters:
            - stepsize
      """

      TT = uu.shape[1]


      stepsizes = []  # list of stepsizes
      costs_armijo = []

      stepsize = stepsize_0

      ns = xx_ref.shape[0]
      ni = uu_ref.shape[0]


      for ii in range(armijo_maxiters):

            # temp solution update

            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))

            xx_temp[:,0] = x0

            for tt in range(TT-1):
                  #uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
                  uu_temp[:,tt] = uu[:,tt] + KKt[:,1:,tt] @(xx_temp[:,tt] - xx[:,tt]) + stepsize * KKt[:,0,tt]
                  xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt], dt)

            # temp cost calculation
            JJ_temp = cst.cost_calculator(xx_temp[:,:], uu_temp[:,:], xx_ref[:,:], uu_ref[:,:], TT)

            stepsizes.append(stepsize)      # save the stepsize
            costs_armijo.append(np.min([JJ_temp, 100*JJ]))    # save the cost associated to the stepsize

            if JJ_temp > JJ  + cc*stepsize*descent_arm:
                  # update the stepsize
                  stepsize = beta*stepsize

            else:
                  print('Armijo stepsize = {:.3e}'.format(stepsize))
                  break
            
            if ii == armijo_maxiters -1:
                  print("WARNING: no stepsize was found with armijo rule!")
                  stepsize = 0
            
      ############################
      # Descent Plot
      ############################

      if plot:

            steps = np.linspace(0,stepsize_0,int(3e1))
            costs = np.zeros(len(steps))

            for ii in range(len(steps)):

                  step = steps[ii]

                  # temp solution update

                  xx_temp = np.zeros((ns,TT))
                  uu_temp = np.zeros((ni,TT))

                  xx_temp[:,0] = x0

                  for tt in range(TT-1):
                        #uu_temp[:,tt] = uu[:,tt] + step*deltau[:,tt]
                        uu_temp[:,tt] = uu[:,tt] + KKt[:,1:,tt] @(xx_temp[:,tt] - xx[:,tt]) + step * KKt[:,0,tt]
                        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt], dt)
                  
                  # temp cost calculation
                  JJ_temp = cst.cost_calculator(xx_temp[:,:], uu_temp[:,:], xx_ref[:,:], uu_ref[:,:], TT)

                  costs[ii] = np.min([JJ_temp, 100*JJ])


            plt.figure("Armijio")
            plt.clf()

      
            plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
            plt.plot(steps, JJ + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            # plt.plot(steps, JJ - descent*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.plot(steps, JJ + cc*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

            plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

            plt.grid()
            plt.xlabel('stepsize')
            plt.legend()
            plt.draw()
            plt.pause(1e-4)

            plt.show()

      return stepsize

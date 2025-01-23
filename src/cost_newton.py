
#
# Cost 
# Matteo Bonucci, Filippo Samorì, Jacopo Subini
# 01/01/2025
#

import numpy as np
import Dynamics as dyn

ns = dyn.ns # number of states
ni = dyn.ni # number of inputs

# Cost function parameters
QQt = np.diag([1000.0, 1000.0, 10, 10]) # state weights
QQT = np.diag([1000.0, 1000.0, 10, 10]) # terminal state weights

RRt = 10*np.eye(ni) # input wheigths

def cost_calculator(xx ,uu , xx_ref , uu_ref, TT):
  """
    Cost function

    Args
      - xx \in \R^2 state vector at iteration kk
      - xx_ref \in \R^2 state reference vector at iteration kk

      - uu \in \R^1 input vector at iteration kk
      - uu_ref \in \R^2 input reference vector at iteration kk

      - TT time horizon

    Return 
      - cost at xx,uu
  """
  JJ = 0 # initialize the cost

  # calculate the cost
  for tt in range(TT-1):
    temp_cost, _, _, _, _, _ = stagecost(xx[:,tt], uu[:,tt], xx_ref[:,tt], uu_ref[:,tt])
    JJ += temp_cost
  
  temp_cost, _ , _ = termcost(xx[:,-1], xx_ref[:,-1])
  JJ += temp_cost

  return JJ


def stagecost(xx,uu, xx_ref, uu_ref):
  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

      - uu \in \R^1 input at time t
      - uu_ref \in \R^2 input reference at time t


    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """

  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

  lx = QQt@(xx - xx_ref)
  lu = RRt@(uu - uu_ref)

  lxx = QQt
  lxu = np.zeros((ni,ns))
  luu = RRt


  return ll.squeeze(), lx, lu, lxx, luu, lxu

def termcost(xT,xT_ref):
  """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xT \in \R^2 state at time t
      - xT_ref \in \R^2 state reference at time t

    Return 
      - cost at xT,uu
      - gradient of l wrt x, at xT,uu
      - gradient of l wrt u, at xT,uu
  
  """

  xT = xT[:,None]
  xT_ref = xT_ref[:,None]

  llT = 0.5*(xT - xT_ref).T@QQT@(xT - xT_ref)

  lTx = QQT@(xT - xT_ref)

  lTxx = QQT

  return llT.squeeze(), lTx, lTxx
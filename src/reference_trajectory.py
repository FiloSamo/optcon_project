

import numpy as np
import Dynamics as dyn


def gen(tf, dt, ns, ni):

      TT = int(tf/dt)
      
      theta1_0 = 0
      theta1_T = 180
      theta2_0 = 0
      theta2_T = 0

      
      xx_ref = np.zeros((ns, TT))
      uu_ref = np.zeros((ni, TT))
    

      xx_ref[0,int(TT/2):] = np.full(int(TT/2),np.deg2rad(theta1_T))
      
   
      return xx_ref, uu_ref
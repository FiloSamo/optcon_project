#
# Animation 
# Matteo Bonucci, Filippo Samorì, Jacopo Subini
# 01/01/2025
#

from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

def animate(xx, xx_ref, dt):
      """
      Animates the pendolum dynamics
      input parameters:
            - Optimal state trajectory xx_star
            - Reference trajectory xx_ref
            - Sampling time dt
      oputput arguments:
            None
      """

      TT = xx.shape[1]

      # Set up the figure and axis for the animation
      fig, ax = plt.subplots()
      ax.set_xlim(-3, 3)  # adjust limits as needed for pendulum's reach
      ax.set_ylim(-3, 3)
      ax.set_aspect("equal")

      # Plot elements
      Base = ax.plot(0,0,'x-', color="blue",  markersize=20)
      link_1, = ax.plot([], [], 'o-', color="blue", markersize=10)
      link_2, = ax.plot([], [], 'o-', color="red", markersize=10)
      trajectory, = ax.plot([], [], 'o-', color="black", markersize=0.5)
      reference_line, = ax.plot([], [], '--', markersize=0.1, color="green", label="Reference Path")
      time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

      ax.set_title("2 dof robotic arm")
      ax.set_xlabel("X position")
      ax.set_ylabel("Y position")

      # Initial setup function for the animation
      def init():
            link_1.set_data([], [])
            link_2.set_data([], [])
            trajectory.set_data([], [])
            reference_line.set_data([], [])
            time_text.set_text('')
            return  reference_line, trajectory, link_1, link_2,  time_text

      # Update function for each frame of the animation
      def update(frame):
            # Link 1
            x_1 = np.sin(xx[0, frame])  # assuming xx_star[0] is angle theta
            y_1 = -np.cos(xx[0, frame])

            # Link 2
            x_2 = np.sin(xx[0, frame])+np.sin(xx[0, frame]+xx[1, frame])
            y_2 = -np.cos(xx[0, frame])-np.cos(xx[0, frame]+xx[1, frame])

            # Reference trajectory
            x_t = np.sin(xx_ref[0, :frame])+np.sin(xx_ref[0, :frame]+xx_ref[1, :frame])
            y_t = -np.cos(xx_ref[0, :frame])-np.cos(xx_ref[0, :frame]+xx_ref[1, :frame])
            trajectory.set_data(x_t,y_t)

            x_r = np.sin(xx_ref[0, :TT])+np.sin(xx_ref[0, :TT]+xx_ref[1, :TT])
            y_r = -np.cos(xx_ref[0, :TT])-np.cos(xx_ref[0, :TT]+xx_ref[1, :TT])

            reference_line.set_data(x_r, y_r)
            
            # Update pendulum link
            link_1.set_data([0, x_1], [0, y_1])
            link_2.set_data([x_1, x_2], [y_1, y_2])

            # Update time text
            time_text.set_text(f'time = {frame*dt:.3f}s')

            return reference_line, trajectory, link_1, link_2,  time_text
      # Create the animation
      ani = FuncAnimation(fig, update, frames=TT, init_func=init, blit=True, interval=0.001)
      # Display the animation
      plt.show()
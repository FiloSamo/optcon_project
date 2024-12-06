#Animation 

from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

def animate(xx, dt):
      """
      Animates the pendolum dynamics
      input parameters:
            - Optimal state trajectory xx_star
            - Reference trajectory xx_ref
            - Sampling time dt
      oputput arguments:
            None
      """
	# commento
      TT = xx.shape[1]

      # Set up the figure and axis for the animation
      fig, ax = plt.subplots()
      ax.set_xlim(-3, 3)  # adjust limits as needed for pendulum's reach
      ax.set_ylim(-3, 3)
      ax.set_aspect("equal")

      # Plot elements
      link_1, = ax.plot([], [], 'o-', color="blue", markersize=10)
      link_2, = ax.plot([], [], 'o-', color="red", markersize=10)
      # reference_line, = ax.plot([], [], 'o--', lw=2, color="green", label="Reference Path")
      time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

      ax.set_title("2 dof robotic arm")
      ax.set_xlabel("X position")
      ax.set_ylabel("Y position")

      # Initial setup function for the animation
      def init():
            link_1.set_data([], [])
            link_2.set_data([], [])
            time_text.set_text('')
            return link_1, link_2, time_text

      # Update function for each frame of the animation
      def update(frame):
            # Link 1
            x_1 = np.sin(xx[0, frame])  # assuming xx_star[0] is angle theta
            y_1 = -np.cos(xx[0, frame])

            # Link 2
            x_2 = np.sin(xx[0, frame])+np.sin(xx[0, frame]+xx[1, frame])
            y_2 = -np.cos(xx[0, frame])-np.cos(xx[0, frame]+xx[1, frame])

            # Update pendulum link
            link_1.set_data([0, x_1], [0, y_1])
            link_2.set_data([x_1, x_2], [y_1, y_2])

            # Update time text
            time_text.set_text(f'time = {frame*dt:.3f}s')

            return link_1, link_2, time_text

      # Create the animation
      ani = FuncAnimation(fig, update, frames=TT, init_func=init, blit=True, interval=0.001)

      # Display the animation
      plt.show()

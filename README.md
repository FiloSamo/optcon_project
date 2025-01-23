# Optimal Control Project

## Description

The `main_newton.py` script is the entry point of the Optimal Control Project. It implements the Newton's-like method to solve the optimal control problem for an underactuated 2-DOF robotic arm. The script initializes the system dynamics, sets up the cost functions, and iteratively solves for the optimal control inputs that minimize the cost. The results are then used to animate the robotic arm's trajectory and compare it with the reference trajectory.
After finding the optimal trajectory using the Newton's-like method, we compute the Linear Quadratic Regulator (LQR) on the optimal trajectory. The LQR is then used to track the system along this trajectory. Additionally, we implement a Model Predictive Control (MPC) strategy to compare its performance with the LQR in tracking the optimal trajectory.

## Project Structure
- `src/Dynamics.py`: Contains the dynamics of the 2-DOF robotic arm.
- `src/cost_newton.py`: Defines the cost functions used in the optimization.
- `src/Animation.py`: Provides functions to animate the robotic arm's trajectory.
- `src/reference_trajectory.py`: Generates reference trajectories for the optimization.
- `src/solver.py`: Optimization algorithms definition.
- `src/armijo.py`: Armijo's stepsize selection method .
- `src/main.py`: main file.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Usage
1. Start the application:
    ```
    python src/main.py
    ```
2. you can select different trajectories by uncomment only the chosen one in the main.py code.

3. By selecting different flags in the newton_solver() function, you can enable or disable different plots:
    - dynamic_plot : Shows plots of the states and the input during iterations.
    - visu_descent_plot : Shows plots of the armijo algorithm at each iteration.

## Contact
For any questions, please contact us at:

- filippo.samori@studio.unibo.it
- jacopo.subini@studio.unibo.it
- matteo.bonucci@studio.unibo.it
ROS combat simulation engine + tank combat agent

this is a custom ROS stack to simulate live combat dynamics between autonomous robot tanks, using a simple point capture game
each agent has an independent internal representation of the simulated world, derived from an array of semi-realistic sensors

the project depends on ROS Noetic, raylib and Open Dynamics Engine, all of which will need to be available to build the project
The CMakeLists.txt will search root and home directory for raylib--it is very easy to install and build on Linux.

Further development is arranged for use with llama.cpp and LibTorch for ML-based action planning, but this was out of scope for the deadline

The project can be built with catkin_make.

To run the program after building, follow this sequence:
run 'rosrun tanksim tanksim_simulator' to launch the simulator program
in the simulator program, press 'X' on the keyboard, as many times as desired, to create tank agents that can be linked to by controller nodes
then run 'rosrun tanksim tanksim_controller' as many times as desired, in new terminals, up to the number of tanks currently in the simulation.
the program will automatically assign agents to controllers, and they will behave autonomously responding to game instructions provided by the simulation

To change camera modes, press 'Z' on the keyboard. This will swap between first and third person view.
You can also press 'C' to change the agent target of the camera.
This part is a bit janky because it was done in under five minutes.

The whole program, for that matter, is very janky--i did not have much time to complete it.

Agent:
 Completed features
 - 2.5D heightfield SLAM using LIDAR sensors
 - odometry and dead reckoning with GPS+IMU
 - PID control for differential drive tanks, with independent throttle and brake actuators for each side
 - A* global planning for observed slam data
 - vector field histogram (vfh) + reactive potential field (RPF) local planning for obstacle avoidance, using LIDAR data
 - heuristic reactive target acquisition and aiming using PID control, derived from simulated computer vision camera data
 
 Uncompleted features
 - high-level tactical planning through GOAP or similar long-term planner
 - team-tactics through central control node, coordination protocols for sensor data
 - reinforcement learning (libtorch) or genetic algorithm based heuristic tuning for team tactics
 - combat memory and odometry for sighted enemies (enable more tactical thinking in RL based planners)
 - simulated sensor noise, extended Kalman filter implementations
 - local LLM-based tactical planning using llama.cpp

Engine:
 Completed features:
 - simulation engine with raylib and open dynamics, modeling realistic tanks in realistic open terrain
 - multi-robot control in ROS Noetic using dynamic namespace generation for topics
 - custom integrated sensor array, loosely modeling real sensors
 - bespoke entity-component-system for dynamic composition of entities, with hierarchical relationships
 - generic serial sensor/actuator apis, enabling easy expansion
 
 Uncompleted features:
 - better graphics
 - headless mode for training simulations and performance metrics
 - more combat games
 - more diverse obstacles such as trees, cliffs, buildings and water.
 - codebase cleaning
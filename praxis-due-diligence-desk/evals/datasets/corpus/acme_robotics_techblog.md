# Inside Acme Robotics' Engineering (company blog, 2025)

## Our IP position

Acme holds 14 issued US patents and 6 pending applications, concentrated in two
areas: closed-loop actuator control (8 patents) and compliant gripper design
(4 patents). The actuator-control portfolio is the technical moat — it is what
lets our arms run force-feedback loops at 2 kHz on commodity hardware.

## The team

Acme was founded in 2019 by Dr. Lena Ortiz (CEO, previously a research scientist
in the MIT Biomimetics lab) and Raj Patel (CTO, ex-Boston Dynamics controls
lead). The engineering org is 47 people; 12 hold robotics PhDs. Attrition in
2024 was 6%, well below the sector.

## Technology stack

Control software is Rust on a real-time Linux target; the perception stack uses
a fine-tuned vision transformer for grasp-point detection. Simulation runs in
NVIDIA Isaac. Acme publishes its URDF models and a subset of its grasp dataset
under a permissive license.

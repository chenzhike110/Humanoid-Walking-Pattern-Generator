# Humanoid-Walking-Pattern-Generator
Use Traditional Method ZMP and Learning Method PPO to generate Walking Pattern for Humanoid Robot

#### Requirements

- Pinocchio
- gym 
- Pybullet
- Numpy

#### Env

![Screenshot from 2021-07-12 00-45-11](./images/env.png)

- Simple Robot with eight force sensors under the feet

#### ZMP

- 2D inverted pendulum

  ![inverted](./images/inverted.svg)

  There are several differential equations as below

  ![diff](./images/diff.svg)

- 2D linear inverted pendulum

  When ![tau](./images/tau.svg)= 0 and f = Mg/cos![theta](./images/theta.svg),  the Centroid will remain the same height as the pendulum falling. Because f * cos![theta](./images/theta.svg)= Mg .

  ![linear_inverted](./images/linear_inverted.png)

  In the horizontal direction

  ![x](./images/x.svg) 

  given initial state x(0) x'(0) and target state x(t) x'(t) compute used time

  ![t](./images/t.svg)

  orbital energy

  ![e](./images/e.svg)

- Change Feet

  ![change](./images/change.png)
  Given orbital energy

  ![constraint](./images/constraint.svg)


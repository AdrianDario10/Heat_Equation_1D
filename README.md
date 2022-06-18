# Heat Equation 1D

Physics informed neural network (PINN) for the 1D Heat equation

This module implements the Physics Informed Neural Network (PINN) model for the 1D Heat equation. The Heat equation is given by (d/dt - c^2 d^2/dx^2)u = 0, where c is 2. It has an initial condition u(t=0, x) = x**2(2-x). Dirichlet boundary condition is given at x = 0,+2. The PINN model predicts u(t, x) for the input (t, x).

It is based on hysics informed neural network (PINN) for the 1D Wave equation on https://github.com/okada39/pinn_wave

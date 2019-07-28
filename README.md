# PDE_tensorflow
Partial Differential Equations, using Tensorflow.

Example is a numerical solution (Euler's method) of wave equation (second order Partial Differencian Equation - PDE). It's quite hard to model PDEs (and computationary expensive?), that's why PDE was transformed into ODE (oridinary differential equation), which only has one differentiable variable - time. And can be easily computed with Euler's method.

Transformation of PDE to ODE is done with Laplacian transformation. In particular, discrete Laplacian operator (very rough approximation of second derivative) was used. More precisely isotropic discrete Laplacian operator. Isotropic discrete Laplacian operator is just a 3x3 matrix (convolutional kernel) in our case.

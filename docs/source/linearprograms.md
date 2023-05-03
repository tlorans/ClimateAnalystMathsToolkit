## Linear Programming 

Optimization programs have constraints and objective function. The objective function is the function you want to minimize. In linear programs (LP), the objective function is linear, and the constraints as well. 

Using a particular solver, requires a particular form. We need to undersand how to manipulate a linear program in order to fit the form required by solvers.

### Linear and Affine Functions

Let's first define what linear and affine functions are.

A function $f(x_1,..., x_m)$ is linear in the variables $x_1, ..., x_m$ if there exists constants $a_1, ... , a_m$ such that:

\begin{equation}
f(x_1,..., x_m) = a_1x_1 + ... + a_mx_m =
a^Tx
\end{equation}

This is basically a dot product. 

A function $f(x_1,..., x_m)$ is affine in the variables $x_1, ..., x_m$ if there exist constants $b$, $a_1, ... , a_m$ such that:


\begin{equation}
f(x_1,..., x_m) = a_0 + a_1x_1 + ... + a_mx_m =
a^Tx + b
\end{equation}

This is basically a dot product plus a constant. 

### The Linear Program

### Standard Form

### Transformation Tricks


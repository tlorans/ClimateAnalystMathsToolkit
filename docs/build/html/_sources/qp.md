## Quadratic Programming

Quadratic programming is an optimization problem with a quadratric objective function. Portfolio optimization is a well-known example of QP, as the objective function to minimize is made quadratric by the portfolio's variance.

### Quadratic Forms

As we have seen in the previous part, the general form of linear functions is:

\begin{equation}
c_1x_1 + ... + c_nx_n = c^Tx
\end{equation}

with $c_i$ are parameters and $x_i$ are variables.

On the other side, the general form of quadratic functions is:
\begin{equation}
q_{11}x^2_1 + q_{12}x_1x_2 + ... + q_{nn}x^2_n
= 
\begin{bmatrix}
x_1 \\ \cdots \\ x_n
\end{bmatrix}^T
\begin{bmatrix}
q_{11} & \cdots & q_{1n}\\
\vdots & \ddots & \vdots \\
q_{n1} & \cdot & q_{nn}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
\vdots \\
x_n
\end{bmatrix}
= x^TQx
\end{equation}

with $q_{ij}$ are parameters and $x_i$ are variables.

### Standard Form

Quadratric program (QP) is like an LP, but with a quadratric objective function:

\begin{equation*}
\begin{aligned}
& x^* = 
&& argmin \; x^TPx + q^Tx + r\\
& \text{subject to}
& & Ax \leq b\\
\end{aligned}
\end{equation*}

### Portfolio Optimization
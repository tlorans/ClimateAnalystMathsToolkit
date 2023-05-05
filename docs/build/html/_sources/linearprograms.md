## Linear Programming 

With the previous part about matrices and matrices manipulation, we already know enough of linear algebra to introduce linear programming.

Optimization programs have constraints and objective function. The objective function is the function you want to minimize. In linear programs (LP), the objective function is linear, and the constraints as well. 

Using a particular solver, requires a particular form. We need to undersand how to manipulate a linear program in order to fit the form required by solvers.

### A First Example

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

We can also combine several linear or affine functions:

\begin{equation}
\begin{matrix}
a_{11}x_1 + & \cdots + & a_{1n}x_n + b_1 \\
a_{21}x_1 + & \cdots + & a_{2n}x_n + b_2 \\
\vdots & \vdots & \vdots \\
a_{m1}x_1 + & \cdots + & a_{mn}x_n + b_m 
\end{matrix} \rightarrow
\begin{bmatrix}
a_{11} & \cdots & a_{1n} \\
\vdots & \ddots & \vdots \\
a_{m1} & \cdots & a_{mn}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
\vdots \\
x_n
\end{bmatrix}
+ 
\begin{bmatrix}
b_1 \\
\vdots \\
b_m
\end{bmatrix}
\end{equation}

Which can be written as $Ax + b$.

### Standard Form

A linear program (LP) is an optimization model with:

- real-valued variables ($x \in \mathbb{R}^n$)
- affine objective function ($c^Tx + d$), min or max
- constraints can be:
    - affine equations ($Ax = b$)
    - affine inequalities ($Ax \leq b$ or $Ax \geq b$)
    - a combination of affine equations or inequalities
- individual variables can have:
    - box constraints ($p \leq x_i$, or $p \leq x_i \leq q$)
    - no constraints ($x_i$ is unconstrainted)

Every LP can be put in the form:

### Transformation Tricks

#### Converting Min to Max or Vice Versa

#### Reversing Inequalities 

#### Equalities to Inequalities 

#### Inequalities to Equalities 

#### Unbounded to Bounded

#### Bounded to Nonnegative


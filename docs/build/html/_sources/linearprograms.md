## Linear Programming 

With the previous part about matrices and matrices manipulation, we already know enough of linear algebra to introduce linear programming.

Optimization programs have constraints and objective function. The objective function is the function you want to minimize. In linear programs (LP), the objective function is linear, and the constraints as well. 

Using a particular solver, requires a particular form. We need to undersand how to manipulate a linear program in order to fit the form required by solvers.

### A First Example

Let's make a first toy example to illustrate optimization modelling, with the Top Brass example:

- The Top Brass Company makes trophies for athletic leagues. The company produces trophies for footabll and soccer championship.
- Each football trophy has a wood base, an engraved plaque, a large brass football on top and returns $12 in profit.
- Soccer throphies are similar, except that a brass soccer ball is on top, and the unit profit is only $9.
- The football has an asymmetric shape and then the base requires 4 board feet of woord.
- The soccer base requires only 2 board feet. 
- At the moment there are 1000 bass footballs in stock, 1500 soccer balls, 1750 plaques and 4800 board feet of wood.

Based on these information, what trophies should be produced from these supplies to maximize total profit, assuming that all that are made can be sold?

We have the following recipe for building each trophy:
|   | Wood  | Plaques  | Footballs  | Soccer Balls  | Profit |
|---|---|---|---|---|---|
| football  |  4 ft |  1 | 1  | 0  | $12 |
|  soccer | 2 ft  |  1 | 0  |  1 |  $9 |

And the following quantity of each ingredient in stock:
|   | Wood  | Plaques  | Footballs  | Soccer Balls  | 
|---|---|---|---|---|
| stocks  |  4800 ft |  1750 | 1000  | 1500  | 

To formulate this problem as an optimization problem, we need to find three components: (i) the decision variables; (ii) the constraints; (iii) the objective function.

1. Decision variables:
    - $f$: number of football trophies built
    - $s$: number of soccer trophies built

2. Constraints:
    - $4f + 2s \leq 4800$ (wood budget)
    - $f + s \leq 1750$ (plaque budget)
    - $0 \leq f \leq 1000$ (football budget)
    - $0 \leq s \leq 1500$ (soccer ball budget)

3. Objective function:
    - Maximize $12f + 9s$ (profit)

The optimization form is:

\begin{equation*}
\begin{aligned}
& f^*, s^* = 
&& argmax \; 12f + 9s\\
& \text{subject to}
& & 4f + 2s \leq 4800\\
& & & f + s \leq 1750 \\
&&&  0 \leq f \leq 1000 \\
&&& 0 \leq s \leq 1500
\end{aligned}
\end{equation*}

This is an example of a linear program (LP), a type of optimization model.
In that model, you have decision variables ($f$ and $s$) and parameters (the rest).

A more generic way to write this problem is:

\begin{equation*}
\begin{aligned}
& f^*, s^* = 
&& argmax \; c_1f + c_2s\\
& \text{subject to}
& & a_{1,1}f + a_{1,2}s \leq b_1\\
& & & a_{2,1}f + a_{2,2}s \leq b_2 \\
&&&  l_1 \leq f \leq u_1 \\
&&& l_2 \leq s \leq u_2
\end{aligned}
\end{equation*}

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

\begin{equation*}
\begin{aligned}
& x^* = 
&& argmax \; c^Tx\\
& \text{subject to}
& & Ax \leq b\\
& & & x \geq 0 \\
\end{aligned}
\end{equation*}

This is the standard form of a LP.

Let's make an illustration with the Top Brass example:


\begin{equation*}
\begin{aligned}
& f^*, s^* = 
&& argmax \; \begin{bmatrix}12 \\
9\end{bmatrix}^T \begin{bmatrix}f \\ s\end{bmatrix}\\
& \text{subject to}
& & \begin{bmatrix}4 & 2 \\
1 & 1  \\
1 & 0 \\
0 & 1 \end{bmatrix} \begin{bmatrix}f \\ s\end{bmatrix}\leq \begin{bmatrix} 4800 \\
1750 \\
1000 \\
1500 \end{bmatrix}\\
& & & \begin{bmatrix} f \\
s\end{bmatrix} \geq 0\\
\end{aligned}
\end{equation*}

### Transformation Tricks

To convert an initial problem to the standard form, there are some transformation tricks.
#### Converting Min to Max or Vice Versa

You can convert a minimization problem to a maximization problem or vice verse (by taking the negative):

\begin{equation}
    \min_x
 f(x) =
  - \max_x
  (-f(x))\end{equation}
#### Reversing Inequalities 

To reverse inequalities, you can simply flip the sign:

\begin{equation}
Ax \leq b \Leftrightarrow (-A)x \geq (-b)
\end{equation}

#### Equalities to Inequalities 

To convert equalities to inequalities, you can double up:

\begin{equation}
f(x) = 0 \Leftrightarrow f(x) \geq 0  \text{ and } f(x) \leq 0
\end{equation}

#### Inequalities to Equalities 

To convert inequalities to equalities, you can add a slack:

\begin{equation}
f(x) \leq 0 \Leftrightarrow f(x) + s = 0 \text{ and } s \geq 0
\end{equation}

#### Unbounded to Bounded

You can change an unbounded to bounded formulation, by adding a difference term:

\begin{equation}
x \in \mathbb{R} \Leftrightarrow u \geq 0, \; v \geq 0 \text{ and } x = u-v
\end{equation}

#### Bounded to Unbounded



#### Bounded to Nonnegative


## Systems of Linear Equations 

In this part, we are going to see how to solve a system of linear equations with linear algebra. 

This is particularly useful to understand how an optimization problem can be modelled in matrix forms and these technics are especially useful for estimating emissions factors with input-output matrices.

### Gaussian Elimination

First of all, let's recall that a linear system of equations such as:

\begin{equation}
\begin{matrix}
-3x_1 + 2x_2 - x_3 = - 1 \\
6x_1 - 6x_2 + 7x_3 = - 7 \\
3x_1 - 4x_2 + 4x_3 = -6
\end{matrix}
\end{equation}

can be written in matrix form as:

\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1 \\
6 & -6 & 7 \\
3 & -4 & 4
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
= \begin{bmatrix}
- 1 \\ 
-7 \\
-6
\end{bmatrix}
\end{equation}

Or simply $Ax = b$. 

First, let's use the `sympy` package to declare our system of linear equations in Python:

```Python
import sympy as sp

x1, x2, x3 = sp.symbols('x1 x2 x3')
symbolic_vars = [x1, x2, x3]

# note that we've moved to representation with each equation = 0
equations = [
    -3 * x1 + 2 * x2 - x3 + 1,
    6 * x1 - 6 * x2 + 7 * x3 + 7,
    3 * x1 - 4 *x2 + 4*x3 + 6
]
```

We now can use the function `linear_eq_to_matrix` from `sympy` in order to get the matrix forms of $A$ and $b$:

```Python
A, b = sp.linear_eq_to_matrix(equations, symbolic_vars)
```

The Gaussian elimination is the standard algorithm to solver a system of linear equations. The first step is to form an augmented matrix by combining the matrix $A$ and the column vector $b$:

\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1 & -1 \\
6 & -6 & 7 & -7 \\
3 & -4 & 4 & -6
\end{bmatrix}
\end{equation}

Let's create the augmented matrix in Python:

```Python
import numpy as np

augmented_A = np.asarray(A.col_insert(len(symbolic_vars), b), dtype=np.float32)
augmented_A
```

Which gives us:
```
array([[-3.,  2., -1., -1.],
       [ 6., -6.,  7., -7.],
       [ 3., -4.,  4., -6.]], dtype=float32)
```

We need to apply row reduction with this augmented matrix. Operations allowed are:

1. Interchange the order of the rows
2. Multiply any row by a constant
3. Add a multiple of one row to another row

The objective is to convert the matrix $A$ to an upper-triangular form and use this new form to solve for the unknowns $x$.

First, we can multiply the first row by 2 and add it to the second row:

\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1 & -1 \\
6 - 6& -6 + 4 & 7 - 2 & -7 - 2 \\
3 & -4 & 4 & -6
\end{bmatrix}

= \begin{bmatrix}
- 3 & 2 & -1 & -1 \\
0 & -2 & 5 & -9 \\
3 & -4 & 4 & -6
\end{bmatrix}
\end{equation}

We can also add the first row the third row:

\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1 & -1 \\
0 & -2 & 5 & -9 \\
3 - 3 & -4 + 2 & 4 - 1 & -6 -1
\end{bmatrix} =
\begin{bmatrix}
- 3 & 2 & -1 & -1 \\
0 & -2 & 5 & -9 \\
0 & -2 & 3 & -7
\end{bmatrix}
\end{equation}

We then can multiply the second row by -1 and add it to the third row:

\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1 & -1 \\
0 & -2 & 5 & -9 \\
0 & -2 + 2 & 3 -5 & -7 + 9
\end{bmatrix} =
\begin{bmatrix}
- 3 & 2 & -1 & -1 \\
0 & -2 & 5 & -9 \\
0 & 0 & -2 & 2
\end{bmatrix}
\end{equation}

Our original $A$ matrix has now been converted to an upper triangular matrix. The new corresponding system of linear equations is:

\begin{equation}
\begin{matrix}
-3 x_1 + 2x_2 - x_3 = -1 \\
-2x_2 + 5x_3 = -9 \\
-2x_3 = 2
\end{matrix}
\end{equation}

Let's proceed to these rows operations in Python:

```Python
augmented_A[1] += augmented_A[0] * 2
augmented_A[2] += augmented_A[0]
augmented_A[2] += augmented_A[1] * -1

augmented_A
```

You can now use back substitution to solve these equations:

\begin{equation}
\begin{matrix}
x_3 = - 1 \\
x_2 = - \frac{1}{2}(-9 - 5x_3) = 2 \\
x_1 = - \frac{1}{3}(-1 + x_3 - 2x_2) = 2
\end{matrix}
\end{equation}

And you get the solution:

\begin{equation}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
= \begin{bmatrix}
2 \\
2 \\
- 1
\end{bmatrix}
\end{equation}

### Reduced Row Echelon Norm

### Computing Inverses 

### Elementary Matrices

### LU Decomposition

### Solving (LU)x = b
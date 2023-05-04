## Systems of Linear Equations 

In this part, we are going to see how to solve a system of linear equations with linear algebra. 

This part is particularly useful to understand the input-output modelling framework for estimating emissions factors. We will progressively go up to the computation of inverses. Theoretically, solving the Input-Output model involves the computation of the matrix inverse $A^{-1}$ to solve a sytem such as $Ax = b$.

We will then see the concept of LU decomposition and how to use it to solve $(LU)x = b$, which is substantially faster in practice, especially for large matrices such as the input-output matrices.

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

Which gives us our upper triangular matrix:
```
array([[-3.,  2., -1., -1.],
       [ 0., -2.,  5., -9.],
       [ 0.,  0., -2.,  2.]], dtype=float32)
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

Let's implement the back substition in Python:

```Python
def back_substitution(M, syms):
    # symbolic variable index
    for i, row in reversed(list(enumerate(M))):
        # create symbolic equation
        eqn = -M[i][-1]
        for j in range(len(syms)):
            eqn += syms[j] * row[j]

        # solve symbolic expression and store variable
        syms[i] = sp.solve(eqn, syms[i])[0]

    # return list of evaluated variables
    return syms

back_substitution(augmented_A, symbolic_vars)
```

And you get the expected results:
```
[2.00000000000000, 2.00000000000000, -1.00000000000000]
```

### Reduced Row Echelon Form

We can continue the row elimination procedure of Gaussian elimination to bring a matrix to what is called a reduced row echelon form, denoted as $rref(A)$ for the matrix $A$. 

The reduced row echelon form of a matrix has:
- 1 as the first nonzero entry in every row
- all the entries below and above this one are zero
- any zero rows occur at the bottom of the matrix

Let's consider another example:

\begin{equation}
A =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
4 & 5 & 6 & 7 \\
6 & 7 & 8 & 9
\end{bmatrix}
\end{equation}

We can first multiply the first row by -4 and add it to the second row:


\begin{equation}
\begin{bmatrix}
1 & 2 & 3 & 4 \\
4 -4 & 5 - 8 & 6 - 12 & 7 - 16 \\
6 & 7 & 8 & 9
\end{bmatrix} =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & -3 & -6 & -9 \\
6 & 7 & 8 & 9
\end{bmatrix}
\end{equation}

We can then multiply the first row by -6 and add it to the third row:


\begin{equation}
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & -3 & -6 & -9 \\
6 - 6 & 7 - 12 & 8 - 18 & 9 - 24
\end{bmatrix} =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & -3 & -6 & -9 \\
0 & -5 & -10 & -15
\end{bmatrix}
\end{equation}

Now we can multiply the second row by -2 and add it to the third row:

\begin{equation}

\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & -3 & -6 & -9 \\
0 & -5 + 6  & -10 + 12 & -15 + 18
\end{bmatrix} =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & -3 & -6 & -9 \\
0 & 1  & 2 & 3
\end{bmatrix}
\end{equation}

We can also multiply the third row by 4 and add it to the second row:

\begin{equation}
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & -3 + 4 & -6 + 8 & -9 + 12 \\
0 & 1  & 2 & 3
\end{bmatrix} =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & 1 & 2 & 3 \\
0 & 1  & 2 & 3
\end{bmatrix} 
\end{equation}

We can multiply the second row by -1 and add it to the third column:

\begin{equation}
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & 1 & 2 & 3 \\
0 & 1 - 1  & 2 - 2 & 3 - 3
\end{bmatrix} =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & 1 & 2 & 3 \\
0 & 0  & 0 & 0
\end{bmatrix}
\end{equation}

And finally, we want to multiply the second row by -2 and add it to the first row:

\begin{equation}
\begin{bmatrix}
1 & 2 - 2 & 3 - 4 & 4 - 6 \\
0 & 1 & 2 & 3 \\
0 & 0  & 0 & 0
\end{bmatrix}=
\begin{bmatrix}
1 & 0 & -1 & -2 \\
0 & 1 & 2 & 3 \\
0 & 0  & 0 & 0
\end{bmatrix}
\end{equation}

Thus:

\begin{equation}
rref(A) = \begin{bmatrix}
1 & 0 & -1 & -2 \\
0 & 1 & 2 & 3 \\
0 & 0  & 0 & 0
\end{bmatrix}
\end{equation}

In that case, we say that $A$ has two pivot columns. That is, two columns that contain a pivot position with a one in $rref(A)$.

The reduced row echelon form of a matrix $A$ is unique.

Furthermore, if $A$ is a square invertible matrix, then $rref(A)$ is the identity matrix $I$.

### Computing Inverses 

We can compute the matrix inverse by converting an invertible matrix to reduced row echelon form. For a given matrix $A$, let's consider the following equation:

\begin{equation}
AA^{-1} = I
\end{equation}

With the unknown inverse $A^{-1}$. 

We denote the columns of $A^{-1}$ as $a_1{-1}$, $a_2^{-1}$, etc. The matrix $A$ multiplying the first column of $A^{-1}$ is the equation:

\begin{equation}
Aa_1^{-1} = e_1
\end{equation}

With:

\begin{equation}
e_1 = \begin{bmatrix}
1 & 0 & \cdots & 0
\end{bmatrix}^T
\end{equation}

And where $e_1$ is the first column of the identity matrix. The general form is:

\begin{equation}
Aa^{-1}_i = e_i
\end{equation}

For $i = 1, ... , n$.

To compute the inverse, the method is then to do row reduction on an augmented matrix which attaches the identity matrix to $A$. We find $A^{-1}$ by continuing the elimination until we obtain $rref(A) = I$.

Let's go back to the first example we use for illustrating Gaussian Elimination:

\begin{equation}
A = \begin{bmatrix}
- 3 & 2 & -1 \\
6 & -6 & 7 \\
3 & -4 & 4
\end{bmatrix}
\end{equation}

We have the augmented matrix $A$ (with the identity matrix $I$ attached):

\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1 & 1 & 0 & 0\\
6 & -6 & 7 & 0 & 1 & 0 \\
3 & -4 & 4 & 0 & 0 & 1
\end{bmatrix}
\end{equation}

We can first add the first row to the third row:
\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1 & 1 & 0 & 0\\
6 & -6 & 7 & 0 & 1 & 0 \\
3 - 3 & -4 + 2 & 4 - 1 & 0 + 1 & 0 & 1
\end{bmatrix}
= \begin{bmatrix}
- 3 & 2 & -1 & 1 & 0 & 0\\
6 & -6 & 7 & 0 & 1 & 0 \\
0 & -2 & 3 & 1 & 0 & 1
\end{bmatrix}
\end{equation}

Then multiply the first row by two and add it to the second row:
\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1 & 1 & 0 & 0\\
6 - 6 & -6 + 4 & 7 - 2 & 0 + 2 & 1 & 0 \\
0 & -2 & 3 & 1 & 0 & 1
\end{bmatrix}
= \begin{bmatrix}
- 3 & 2 & -1 & 1 & 0 & 0\\
0 & -2 & 5 & 2 & 1 & 0 \\
0 & -2 & 3 & 1 & 0 & 1
\end{bmatrix}
\end{equation}

We can multiply the second row by -1 and add it to the third row:
\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1 & 1 & 0 & 0\\
0 & -2 & 5 & 2 & 1 & 0 \\
0 & -2 + 2 & 3 - 5 & 1 - 2 & 0 - 1 & 1
\end{bmatrix} = 
\begin{bmatrix}
- 3 & 2 & -1 & 1 & 0 & 0\\
0 & -2 & 5 & 2 & 1 & 0 \\
0 & 0 & -2 & -1 & - 1 & 1
\end{bmatrix} 
\end{equation}

We can add the second row to the first row:
\begin{equation}
\begin{bmatrix}
- 3 & 2 - 2 & -1 + 5 & 1 + 2 & 0 + 1 & 0\\
0 & -2 & 5 & 2 & 1 & 0 \\
0 & 0 & -2 & -1 & - 1 & 1
\end{bmatrix}=
 \begin{bmatrix}
- 3 & 0 & 4 & 3 & 1 & 0\\
0 & -2 & 5 & 2 & 1 & 0 \\
0 & 0 & -2 & -1 & - 1 & 1
\end{bmatrix}
\end{equation}

We can multiply the third row by two and add it to the first row:

\begin{equation}
\begin{bmatrix}
- 3 & 0 & 4 - 4 & 3 -2 & 1 - 2 & 0 + 2\\
0 & -2 & 5 & 2 & 1 & 0 \\
0 & 0 & -2 & -1 & - 1 & 1
\end{bmatrix}
=
\begin{bmatrix}
- 3 & 0 & 0 & 1 & -1 & 2\\
0 & -2 & 5 & 2 & 1 & 0 \\
0 & 0 & -2 & -1 & - 1 & 1
\end{bmatrix}
\end{equation}

We can then multiply the third row by $\frac{5}{2}$ and add it to the second row:

\begin{equation}
\begin{bmatrix}
- 3 & 0 & 0 & 1 & -1 & 2\\
0 & -2 & 5 - \frac{10}{2} & 2 - \frac{5}{2} & 1 - \frac{5}{2}  & 0 + \frac{5}{2}\\
0 & 0 & -2 & -1 & - 1 & 1
\end{bmatrix} = 
\begin{bmatrix}
- 3 & 0 & 0 & 1 & -1 & 2\\
0 & -2 & 0 & - \frac{1}{2} & - \frac{3}{2}  & \frac{5}{2}\\
0 & 0 & -2 & -1 & - 1 & 1
\end{bmatrix}
\end{equation}

Finally, we can multiply the first row by $-\frac{1}{3}$, the second and third rows by $-\frac{1}{2}$ and we get:

\begin{equation}
\begin{bmatrix}
- 3 \times -\frac{1}{3} & 0 & 0 & 1 \times -\frac{1}{3} & -1 \times -\frac{1}{3} & 2 \times -\frac{1}{3}\\
0 & -2 \times  -\frac{1}{2}& 0 & - \frac{1}{2}\times  -\frac{1}{2}& - \frac{3}{2}  \times  -\frac{1}{2}  & \frac{5}{2}  \times  -\frac{1}{2}\\
0 & 0 & -2  \times  -\frac{1}{2} & -1  \times  -\frac{1}{2} & - 1  \times  -\frac{1}{2} & 1 \times  -\frac{1}{2}
\end{bmatrix}
= \begin{bmatrix}
1 & 0 & 0 & -\frac{1}{3} & \frac{1}{3} & -\frac{2}{3}\\
0 & 1 & 0 & \frac{1}{4} & \frac{3}{4} & -\frac{5}{4}  \\
0 & 0 & 1 & \frac{1}{2} & \frac{1}{2} & -\frac{1}{2}
\end{bmatrix}
\end{equation}

Then we have:

\begin{equation}
A^{-1} = 
\begin{bmatrix}
 -\frac{1}{3} & \frac{1}{3} & -\frac{2}{3}\\
 \frac{1}{4} & \frac{3}{4} & -\frac{5}{4}  \\
 \frac{1}{2} & \frac{1}{2} & -\frac{1}{2}
\end{bmatrix}
\end{equation}

### Elementary Matrices

We can use elementary matrices to implement the row reduction algorithm of Gaussian eliminiation.

These elementary matrices differ from the identity matrix by a single elmentary row operation. 

Let's take again our example with the matrix, with the first stage of the Gaussian elimination algorithm consisting in multiplying the row by two and adds it to the second one:

\begin{equation}
A = \begin{bmatrix}
- 3 & 2 & -1 \\
6 & -6 & 7 \\
3 & -4 & 4
\end{bmatrix} \rightarrow 
\begin{bmatrix}
- 3 & 2 & -1  \\
0 & -2 & 5 \\
3 & -4 & 4 
\end{bmatrix} = 
M_1A
\end{equation}

Where:

\begin{equation}
M_1 = \begin{bmatrix}
1 & 0 & 0 \\
2 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\end{equation}

This matrix $M_1$ multiplies the first row by two and adds the result to the second row (this is why the element two is placed in the column 1, row 2).

We have a second step with:
\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1  \\
0 & -2 & 5 \\
3 & -4 & 4 
\end{bmatrix}\rightarrow 
\begin{bmatrix}
- 3 & 2 & -1  \\
0 & -2 & 5 \\
0 & -2 & 3
\end{bmatrix} = 
M_2M_1A
\end{equation}


Where:

\begin{equation}
M_2 = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
1 & 0 & 1
\end{bmatrix}
\end{equation}

With the number one placed in column one, row three, and the matrix multiplies the first row by one and adds the result to the third row.

Finally, we have:
\begin{equation}
\begin{bmatrix}
- 3 & 2 & -1  \\
0 & -2 & 5 \\
0 & -2 & 3
\end{bmatrix}\rightarrow 
\begin{bmatrix}
- 3 & 2 & -1  \\
0 & -2 & 5 \\
0 & 0 & -2
\end{bmatrix} = 
M_3M_2M_1A
\end{equation}


Where:

\begin{equation}
M_3 = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & -1 & 1
\end{bmatrix}
\end{equation}

With the -1 placed in column two, row three and the matrix multiplies the second row by -1 and adds the result to the third row.

Because the result is an upper triangular matrix, we have found that:

\begin{equation}
M_3M_2M_1A = U
\end{equation}
### LU Decomposition

We have found in the previous part that:


\begin{equation}
M_3M_2M_1A = U
\end{equation}

If we invert the elementary matrices, we have:

\begin{equation}
A = M^{-1}_1M_2^{-1}M_3^{-1}U
\end{equation}

The matrix $M_1$ multiplies the first row by two and adds it to the second row:

\begin{equation}
M_1 = \begin{bmatrix}
1 & 0 & 0 \\
2 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\end{equation}

To invert this operation, we need to multiply the first row by -2 and add it to the second row:

\begin{equation}
M_1^{-1} = \begin{bmatrix}
1 & 0 & 0 \\
-2 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\end{equation}

Applying the same process, we have:

\begin{equation}
M_2^{-1} = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
-1 & 0 & 1
\end{bmatrix}
\end{equation}

\begin{equation}
M_3^{-1} = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 1 & 1
\end{bmatrix}
\end{equation}

We have then:

\begin{equation}
L = M^{-1}_1M^{-1}_2M^{-1}_3
\end{equation}

\begin{equation}
L = \begin{bmatrix}
1 & 0 & 0 \\
-2 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
-1 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 1 & 1
\end{bmatrix}
= \begin{bmatrix}
1 & 0 & 0 \\
-2 & 1 & 0 \\
-1 & 1 & 1
\end{bmatrix}
\end{equation}

Which is a lower triangular matrix. 

We then have the LU decomposition of $A$ as $A = LU$:

\begin{equation}
 \begin{bmatrix}
- 3 & 2 & -1 \\
6 & -6 & 7 \\
3 & -4 & 4
\end{bmatrix}
= \begin{bmatrix}
1 & 0 & 0 \\
-2 & 1 & 0 \\
-1 & 1 & 1
\end{bmatrix}
\begin{bmatrix}
- 3 & 2 & -1  \\
0 & -2 & 5 \\
0 & 0 & -2
\end{bmatrix}
\end{equation}

### Solving (LU)x = b

LU decomposition we've seen in the previou part is useful when we need to solve $Ax = b$ with large size. With the LU decomposition, we can write:

\begin{equation}
(LU)x = L(Ux) = b
\end{equation}

and:

\begin{equation}
y = Ux
\end{equation}

Then, we can solve:

\begin{equation}
Ly = b
\end{equation}

For $y$ by forward substition, and:

\begin{equation}
Ux = y
\end{equation}

For $x$ with backward substitution. 

For large matrices, solving $(LU)x = b$ is really faster than solving $Ax = b$ directly.

We can illustrate $LUx = b$ with:

\begin{equation}
L = \begin{bmatrix}
1 & 0 & 0 \\
-2 & 1 & 0 \\
-1 & 1 & 1
\end{bmatrix}
\end{equation}


\begin{equation}
U = \begin{bmatrix}
-3 & 2 & -1 \\
0 & -2 & 5 \\
0 & 0 & -2
\end{bmatrix}
\end{equation}

\begin{equation}
b = \begin{bmatrix}
-1  \\
-7  \\
-6
\end{bmatrix}
\end{equation}

With $y = Ux$, we first solve $L_y = b$:

\begin{equation}
\begin{bmatrix}
1 & 0 & 0 \\
-2 & 1 & 0 \\
-1 & 1 & 1
\end{bmatrix}
\begin{bmatrix}
y_1 \\ 
y_2 \\
y_3
\end{bmatrix}
= 
\begin{bmatrix}
-1 \\ 
- 7 \\
- 6
\end{bmatrix}
\end{equation}

Using forward substitution:

\begin{equation}
\begin{matrix}
y_1 = -1 \\
y_2 = -7 + 2 y_1 = - 9 \\
y_3 = -6 + y_1 - y_2 = 2
\end{matrix}
\end{equation}

We can then solve $Ux = y$:

\begin{equation}
\begin{bmatrix}
-3 & 2 & -1 \\
0 & -2 & 5 \\
0 & 0 & -2
\end{bmatrix}
\begin{bmatrix}
x_1 \\ 
x_2 \\
x_3
\end{bmatrix}
= \begin{bmatrix}
- 1 \\
- 9 \\
2
\end{bmatrix}
\end{equation}

We can use back substitution:

\begin{equation}
\begin{matrix}
x_3 = -1 \\
x_2 = - \frac{1}{2}(-9 - 5 x_3) = 2 \\
x_1 = - \frac{1}{3}(-1 - 2x_2 + x_3) = 2
\end{matrix}
\end{equation}

Thus we have found the solution:

\begin{equation}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
= 
\begin{bmatrix}
2 \\
2 \\
- 1
\end{bmatrix}
\end{equation}


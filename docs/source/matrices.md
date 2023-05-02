## Matrices

In this part, we will cover the concept of matrices. We will define matrices, how to operate with, review some particular matrices (identity matrix for example), learn about transposes and inverses, and see the orthogonal and permutation matrices. All of this will be used later in the optimization part. Matrices are particularly important for a climate analyst, from emissions factor estimates with input-output matrices to portfolio optimization problem formulation.

### Matrix Definition

A $m$-by-$n$ matrix is a rectangular array of numbers or maths objects, with $m$ rows and $n$ columns. 

A two-by-two matrix $A$ with two rows and two columns looks like:
\begin{equation}
A = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\end{equation}

The usual library to use in Python for linear algebra is `numpy`. Let's create a simple two-by-two matrix in Python:

```Python
import numpy as np

np.array([["a","b"],["c","d"]])
```

```
array([['a', 'b'],
       ['c', 'd']], dtype='<U1')
```

Column and row matrices are of special importance: these matrices are called vectors. 
We generally denote a column vector as a $n$-by-one vector and the row vector as one-by-$n$.

As an example, a $n = 3$ column vector would be written as:

\begin{equation}
x = \begin{bmatrix}
a \\
b \\
c
\end{bmatrix}
\end{equation}

And a row vector $m = 3$ as:

\begin{equation}
y = \begin{bmatrix}
a & b & c
\end{bmatrix}
\end{equation}

With `numpy`, you can create a row vector as:

```Python
np.array([["a","b","c"]]).shape
```

```
(1, 3)
```

Or a column vector as:

```Python
np.array([["a"],
          ["b"],
          ["c"]
          ]).shape
```

```
(3, 1)
```


Finally, a general notation that you will encounter for writing a $m$-by-$n$ matrix $A$ is:

\begin{equation}
A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\end{equation}

With $a_{ij}$ the element of $A$ in the $i$th row and the $j$th column.

### Matrix Addition and Multiplication 

We can add two matrices only if they have the same dimension $m$ and $n$. We proceed the addition element by element:

\begin{equation}
\begin{bmatrix}
a & b \\
c & d 
\end{bmatrix}
+ 
\begin{bmatrix}
e & f \\
g & h
\end{bmatrix}
= 
\begin{bmatrix}
a + e & b + f \\
c + g & d + h
\end{bmatrix}
\end{equation}

Let's test it in Python:

```Python
A = np.array([[1,2],
          [3,4]])

B = np.array([[5,6],
          [7,8]])

A + B
```

```
array([[ 6,  8],
       [10, 12]])
```


### Special Matrices

### Matrix Transpose

### Inner and Outer Products 

### Inverse Matrix 

### Orthogonal Matrices 

### Rotation Matrices

### Permutation Matrices
## Matrices

In this part, we will cover the concept of matrices. We will define matrices, how to operate with, review some particular matrices (identity matrix for example), learn about transposes and inverses, and see the orthogonal and permutation matrices. All of this will be used later in the optimization part. Matrices are particularly important for portfolio optimization problem formulation.

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

We can multiply matrices by a scalar. In that case, we just multiply every element of the matrix:

\begin{equation}
k \begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
= \begin{bmatrix}
ka & kb \\
kc & kd
\end{bmatrix}
\end{equation}

Again, in Python it gives:

```Python
A = np.array([[1,2],
          [3,4]])

3 * A
```

```
array([[ 3,  6],
       [ 9, 12]])
```

Matrices can be multiplied by something other than a scalar if the number of columns $n$ of the left matrix equals the number of rows $m$ of the right matrix. 
A matrix $m$-by-$n$ can only be multiplied by another matrix $n$-by-$k$, and the resulting matrix will be of dimension $m$-by-$k$:

\begin{equation}
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\begin{bmatrix}
e & f \\
g & h
\end{bmatrix}
= \begin{bmatrix}
ae + bg & af + bh \\
ce + dg & cf + dh
\end{bmatrix}
\end{equation}

This is how it proceeds:
- The first row of the left matrix is multiplied against and summed with the first column of the right matrix to obtain the element in the first row and the first column of the product matrix ($ae + bg$)
- The first row of the lew matrix is multiplied against and summed the second column of the right matrix to obtain the element in the first row and the second column of the product matrix ($af + bh$)
- The same apply for the second row

We can write a more general formula: element $c_{ij}$ in the product matrix $C$ is obtained by multiplying and summing the elements in row $i$ of the left matrix and in column $j$ of the right matrix:

\begin{equation}
c_{ij} = \sum^n_{k=1}a_{ik}b_{kj}
\end{equation}

We first can test matrix multiplication in Python with two matrices of the wrong dimensions:

```Python
A = np.array([[1,2],
          [3,4]])

B = np.array([[1, 2, 3],
             [4,5,6],
             [7,8,9]])

np.matmul(A, B)
```

The error message is:
```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)
```

In our case, either $A$ should be of dimension two-by-three or $B$ should be of dimension two-by-three in order that the multiplication of the two matrices is possible. Let's modify our previous code:

```Python
A = np.array([[1,2],
          [3,4]])

B = np.array([[1, 2, 3],
             [4,5,6]])

np.matmul(A, B)
```
```
array([[ 9, 12, 15],
       [19, 26, 33]])
```


### Special Matrices

A matrix consisting of all zero elements is called the zero matrix, denoted by $0$. It can be of any size. Multiplication by a zero matrix result in a zero matrix.

Let's see in Python:

```Python
Z = np.zeros((2,2))
A = np.array([[1,2],
          [3,4]])

np.matmul(A,Z) 
```

```
array([[0., 0.],
       [0., 0.]])
```

The identity matrix is a square matrix ($n = m$) with ones on the main diagonal. It is denoted by $I$.
A property of $I$ is the following result, if $A$ and $I$ are square matrices of the same size:

\begin{equation}
AI = IA = A
\end{equation}


It means that multiplication by the identity matrix leaves the matrix unchanged.

Let's see it in action in Python:

```Python
I = np.identity(2)
A = np.array([[1,2],
          [3,4]])
np.matmul(A, I)
```

```
array([[1., 2.],
       [3., 4.]])
```
Zero and identity matrices play the role of the numbers zero and one in matrix multiplication.

Diagonal matrix has its nonzero elements on the diagonal. An example of a two-by-two diagonal matrix is:

\begin{equation}
D = \begin{bmatrix}
d_1 & 0 \\
0 & d_2
\end{bmatrix}
\end{equation}

We can use the `diag` function for creating a diagonal matrix:
```Python
np.diag([1,2])
```

```
array([[1, 0],
       [0, 2]])
```

Upper and lower triangular matrices are square matrix with zero elements below or above the diagonal:

\begin{equation}
U = \begin{bmatrix}
a & b & c \\
0 & d & e \\
0 & 0 & f
\end{bmatrix}
\end{equation}

\begin{equation}
L = \begin{bmatrix}
a & 0 & 0 \\
b & d & 0 \\
c & e & f
\end{bmatrix}
\end{equation}

You can create it in Python by using `tril` and `triu` functions:

```Python
np.tril([1, 2, 3])
```

```
array([[1, 0, 0],
       [1, 2, 0],
       [1, 2, 3]])
```

```Python
np.triu([1, 2, 3])
```

```
array([[1, 2, 3],
       [0, 2, 3],
       [0, 0, 3]])
```



### Transpose Matrix

The transpose of the matrix $A$ is denoted $A^T$ and switches the rows and columns of $A$:

If:


\begin{equation}
A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\end{equation}

Then:

\begin{equation}
A^T  = \begin{bmatrix}
a_{11} & a_{21} & \cdots & a_{m1} \\
a_{12} & a_{22} & \cdots & a_{m2} \\
\vdots & \vdots & \ddots & \vdots \\
a_{1n} & a_{2n} & \cdots & a_{mn}
\end{bmatrix}
\end{equation}

If $A$ is $m$-by-$n$, then $A^T$ is $n$-b-$m$.

As an example:

\begin{equation}
\begin{bmatrix}
a & d \\
b & e \\
c & f \\
\end{bmatrix}^T =
\begin{bmatrix}
a & b & c \\
d & e & f
\end{bmatrix}
\end{equation}

We can get the transpose of a matrix in Python with:

```Python
A = np.array([[1,2],
          [3,4]])

A.T
```

```
array([[1, 3],
       [2, 4]])
```

With transpose matrices, we have the following facts:

\begin{equation}
(A^T)^T = A
\end{equation}

\begin{equation}
(A + B)^T = A^T + B^T
\end{equation}

\begin{equation}
(AB)^T = B^TA^T
\end{equation}

And finally, if $A$ is a square matrix and $A^T = A$, $A$ is called a symmetric matrix.
### Inner and Outer Products 

The inner product, also called the dot product, is a matrix product between a row vector and a column vector. We can obtain a row vector from a column vector with the transpose operator:

\begin{equation}
u^Tv = \begin{bmatrix}
u_1 & u_2 & u_3
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2 \\
v_3
\end{bmatrix} =
u_1v_1 + u_2v_2 + u_3v_3
\end{equation}

In Python (please note that we have created the column vectors as transpose of row vectors, with `.T`): 

```Python
u = np.array([[1, 2, 3]]).T
v = np.array([[4, 5, 6]]).T
u.T @ v
```

When the inner product between two vectors with nonzero elements is zero, we say that these vectors are orthogonal. 

Let's have two formulations of the same example in Python. With `numpy`, you can instantiate your vectors as 1 dimension only, this way:
```Python
u = np.array([1, 2]) 
v = np.array([2, -1]) 
v.shape
```
Then the resulting array is only 1 dimension:
```
(2,)
```
In that case, `numpy` takes in charge to get the proper transpose in case of applying the dot product:

```Python
u.T @ v
```
Will output:
```
0
```
And:
```Python
u @ v
```
Will also output:
```
0
```

If you are creating column vectors, you need to specify the transpose:
```Python
u = np.array([[1, 2]]).T # column vector
v = np.array([[2, -1]]).T # column vector
u @ v
```
If not, you will encounter this error:
```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 1)
```
```Python
u = np.array([[1, 2]]).T # column vector
v = np.array([[2, -1]]).T # column vector
u.T @ v
```
The output is:
```
array([[0]])
```
That is, a one-by-one matrix, ie. a scalar.


The norm of a vector is defined by:

\begin{equation}
||u|| = (u^Tu)^{1/2} = (u^2_1 + u^2_2 + u^2_3)^{1/2}
\end{equation}

If this norm is equal to one, the vector is normalized.

You can use the function `linalg.norm` to get the norm of a vector:
```Python
np.linalg.norm(u)
```
```
2.23606797749979
```

If a set of vectors are mutually orthogonal and normalized, we say that they are orthonormal.

You can also encouter the outer product:

\begin{equation}
uv^T = \begin{bmatrix}
u_1 \\
u_2 \\
u_3
\end{bmatrix}
\begin{bmatrix}
v_1 & v_2 & v_3
\end{bmatrix}
= \begin{bmatrix}
u_1v_1 & u_1v2 & u_1v_3 \\
u_2v_1 & u_2v_2 & u_2v_3 \\
u_3v_1 & u_3v_2 & u_3v_3
\end{bmatrix}
\end{equation}

To obtain the outer product in Python, you must be aware of creating your vectors with two dimensions:

```Python
u = np.array([[1, 2]]).T # column vector
v = np.array([[2, -1]]).T # column vector
u @ v.T
```

```
array([[ 2, -1],
       [ 4, -2]])
```
### Inverse Matrix 

Square matrices can have inverses. A matrix $A$ with an invers is said invertible, and its inverse is denoted by $A^{-1}$. A property of the inverse matrix is:

\begin{equation}
AA^{-1} = A^{-1}A = I
\end{equation}

If we have two invertibles matrices $A$ and $B$, then:

\begin{equation}
(AB)^{-1} = B^{-1}A^{-1}
\end{equation}

And, if $A$ is invertible:

\begin{equation}
(A^T)^{-1} = (A^{-1})^T
\end{equation}

As an illustration of the derivation of the inverse of a two-by-two matrix:

\begin{equation}
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}^{-1}
= \frac{1}{ad-bc} 
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
\end{equation}

Where $ad - bc$ is the definition of the determinant of a two-by-two matrix:

\begin{equation}
det \begin{bmatrix}
a & b \\
c & d
\end{bmatrix} = ad - bc
\end{equation}

The determinants of a two-by-two matrix is then the product of the diagonal ($ad$) minus the product of the off-diagonals ($bc$). Because the determinant is in the denominator in the formula of the derivation of the matrix, a two-by-two matrix is invertible only if $det(A) \neq 0$.

We will see later that a $n$-by-$n$ matrix is invertible if and only if its determinant is nonzero. We will need a more general definition of the determinant.

In Python, we find the inverse of a matrix with the `linalg.inv` function:

```Python
A = np.array([[1,2],
          [3,4]])

np.linalg.inv(A)
```
```
array([[-2. ,  1. ],
       [ 1.5, -0.5]])
```
### Orthogonal Matrices 

An orthogonal matrix is a square matrix $Q$ that satisfies:

\begin{equation}
Q^{-1} = Q^T
\end{equation}

Other properties are:

\begin{equation}
QQ^T = I
\end{equation}

\begin{equation}
Q^TQ = I
\end{equation}

Let's examine a two-by-two example, with $Q$ the orthogonal matrix given by:

\begin{equation}
Q = \begin{bmatrix}
q_{11} & q_{12} \\
q_{21} & q_{22}
\end{bmatrix} =
\begin{bmatrix}
q_1 & q_2
\end{bmatrix}
\end{equation}

that is $q_1$ and $q_2$ are the two-by-one column vectors of $Q$.

Let's check if $Q$ is orthogonal:

\begin{equation}
Q^TQ = \begin{bmatrix}
q_1^T \\
q_2^T
\end{bmatrix}
\begin{bmatrix}
q_1 & q_2
\end{bmatrix}
= \begin{bmatrix}
q_1^Tq_1 & q_1^Tq_2 \\
q_2^Tq_1 & q_2^Tq_2
\end{bmatrix}
\end{equation}

So, if $Q$ is orthogonal, then $Q^TQ = I$ and:

\begin{equation}
q^T_1q_1 = q_2^Tq_2 = 1
\end{equation}

\begin{equation}
q_1^Tq_2 = q_2^Tq_1 = 0
\end{equation}

Which means that the columns of $Q$, $q_1$ and $q_2$, form an orthonormal (mutually orthogonal and normalized) set of vectors. The same apply for the rows.

An example in Python is:

```Python
Q = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1]
])

Q.T @ Q
```
Which gives you the identity matrix:
```
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]])
```

### Permutation Matrices

Permutation matrix is a type of orthogonal matrix that:
- when multiplying on the left, permutes the rows of a matrix
- when multiplying on the right, permutes the columns of a matrix

An example is:

\begin{equation}
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
= \begin{bmatrix}
c & d \\
a & b
\end{bmatrix}
\end{equation}

If we denote the initial order as a set $\{1,2\}$ (ie. the first row in the first place, the second row in the second place), we can denote the order of the permuted matrix as $\{2,1\}$ (we have switched the order of the rows). To obtain the corresponding permutation matrix, we have permuted the corresponding rows of the identify matrix, such as $\{1,2\} \rightarrow \{2,1\}$:

\begin{equation}
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\rightarrow
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\end{equation}

Let's have an example of constructing a permutation matrix in Python, with the goal to switch the rows such as $\{1, 2, 3\} \rightarrow \{2, 1, 3\}$

```Python
I = np.identity(3)
P = np.take(I, [1, 0, 2], axis=0)
P
```
```
array([[0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.]])
```
And let switch the rows of $A$:
```Python
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

P @ A
```

```
array([[4., 5., 6.],
       [1., 2., 3.],
       [7., 8., 9.]])
```
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

A well-known example of QP is portfolio optimization, due to the portfolio variance introduced into the objective function, making it quadratric.


Let's consider a universe of $n$ assets. We have a vector of assets' weights in the portfolio: $x = (x_1, ..., x_n)$.

We have a vector of assets' returns $R = (R_1, ..., R_n)$. 

The return of the portfolio is equal to:

\begin{equation}
R(x) = \sum^n_{i=1}x_iR_i = x^T R
\end{equation}

The expected return of the portfolio is:
\begin{equation}
\mu(x) = \mathbb{E}[R(x)] = \mathbb{E}[x^TR] = x^T \mathbb{E}[R] = x^T \mu
\end{equation}

The expected return of the portfolio is then simply the weighted average of the assets' returns in the portfolio (weighted by their relative weight).

```Python
import numpy as np

x = np.array([0.25, 0.25, 0.25, 0.25])
mu = np.array([0.05, 0.06, 0.08, 0.06])
```
```Python
mu_portfolio = x.T @ mu
print(mu_portfolio)
```

```
0.0625
```

The thing is slightly more complicated with the portfolio's variance. Indeed, you need to take into account the covariance matrix between the assets in the portfolio in order to obtain a proper measure of the variance of the portfolio:

\begin{equation}
\sigma^2(x) = \mathbb{E}[(R(x) - \mu(x))(R(x) - \mu(x))^T]
\end{equation}

\begin{equation}
= \mathbb{E}[(x^TR - x^T\mu) (x^TR - x^T\mu)^T]
\end{equation}

\begin{equation}
= \mathbb{E}[x^T(R-\mu)(R - \mu)^T x]
\end{equation}

\begin{equation}
x^T \mathbb{E}[(R-\mu)(R-\mu)^T]x
\end{equation}

\begin{equation}
= x^T \Sigma x
\end{equation}


In the Markowitz framework, the mean-variance investor considers maximizing the expected return of the portfolio under a volatility constraint (Roncalli, 2023):

\begin{equation*}
\begin{aligned}
& x^* = 
& & argmax & \mu(x) \\
& \text{subject to}
& & 1_n^Tx = 1, \\
&&& \sigma(x) \leq \sigma^*
\end{aligned}
\end{equation*}

Or, equivalently, minimizing the volatility of the portfolio under a return constraint:

\begin{equation*}
\begin{aligned}
& x^* = 
& & argmin & \sigma(x) \\
& \text{subject to}
& & 1_n^Tx = 1, \\
&&& \mu* \leq \mu(x)
\end{aligned}
\end{equation*}

This is the optimization problem of finding the most efficient risk-returns couple mentioned previously, with the portfolio's two moments.

For ease of computation, Markowitz transformed the two original non-linear optimization problems into a quadratic optimization problem. 
Introducing a risk-tolerance parameter ($\gamma$-problem, Roncalli 2013) and the long-only constraint, we obtain the following quadratic problem:

\begin{equation*}
\begin{aligned}
& x^* = 
& & argmin \frac{1}{2} x^T \Sigma x - \gamma x ^T \mu\\
& \text{subject to}
& & 1_n^Tx = 1 \\
& & & 0_n \leq x \leq 1_n
\end{aligned}
\end{equation*}

To solve this problem with Python, we will use the `qpsolvers` library. This library considers the following QP formulation:


\begin{equation*}
\begin{aligned}
& x^* = 
& & argmin \frac{1}{2} x^T P x + q^T x \\
& \text{subject to}
& & Ax = b, \\
&&& Gx \leq h,\\
& & & lb \leq x \leq ub
\end{aligned}
\end{equation*}

We need to find $\{P, q, A, b, G, h, lb, ub\}$. In the previous case, $P = \Sigma$, $q = \gamma \mu$, $A = 1_n^T$, $b = 1$, $lb = 0_n$ and $ub = 1_n$.

To go further, we first need to install the package `qpsolvers`:
```Python
!pip install qpsolvers 
```
We can now define a concrete dataclass with the `MeanVariance` class. This class will require two elements `mu` and `Sigma`. We also define the method `get_portfolio`, requiring a `gamma` parameter to be provided:

```Python
from qpsolvers import solve_qp

@dataclass
class MeanVariance:

  mu: np.array # Expected Returns
  Sigma: np.matrix # Covariance Matrix
  
  def get_portfolio(self, gamma:int) -> Portfolio:
    """QP Formulation"""

    x_optim = solve_qp(P = self.Sigma,
              q = -(gamma * self.mu),
              A = np.ones(len(self.mu)).T, # fully invested
              b = np.array([1.]), # fully invested
              lb = np.zeros(len(self.mu)), # long-only position
              ub = np.ones(len(self.mu)), # long-only position
              solver = 'osqp')

    return Portfolio(x = x_optim, mu = self.mu, Sigma = self.Sigma)
```
This new class will return a `Portfolio` object if we call the `get_portfolio` method with an instantiated object. Let's find several optimum portfolios with various value of $\gamma$, and plot the result:
```Python
test = MeanVariance(mu = mu, Sigma = Sigma)
```

```Python
from numpy import arange

list_gammas = arange(-1,1.2, 0.01)
list_portfolios = []


for gamma in list_gammas:
  list_portfolios.append(test.get_portfolio(gamma = gamma))
```

```Python
import matplotlib.pyplot as plt

returns = [portfolio.get_expected_returns() * 100 for portfolio in list_portfolios]
variances = [portfolio.get_variance() * 100 for portfolio in list_portfolios]

plt.figure(figsize = (10, 10))
plt.plot(variances, returns)
plt.xlabel("Volatility (in %)")
plt.ylabel("Expected Return (in %)")
plt.title("Efficient Frontier")
plt.show()
```
```{figure} efficient_frontier.png
---
name: efficientfrontier
---
Figure: Efficient Frontier
```

This is the well-known efficient frontier. Every portfolios on the efficient frontier (that is, the upper side of this curve) are efficient in the Markowitz framework, depending on the risk-tolerance ($\gamma$ parameter) of the investor.



See wip3 for how to use the model

Implements the following methods:
\begin{algorithm}
  \caption{Alternating Minimization for the R-term SKPD with both L1 and L2 norms}
  \begin{algorithmic}[1]
    \State \textbf{Input:} $y_i$ and $X_i$, $i = 1, \ldots, n$
    \State \textbf{Initialization:} $\hat{A}^{(0)}$ is taken as the top-R left singular vectors of $\sum_{i=1}^n X_i y_i$; with $\hat{X}_i = R(X_i)$ for matrix image or tensor image, $\zeta^{(0)}=\vec{0}$, $z^{(0)}=\vec{0}$ and $(t) = 1, \ldots, T$.
    \For{$t \in \{0, 1, 2, \ldots, T-1\}$}
      \State $\hat{B}^{(t+1)} \gets \min_B \frac{1}{2n} \sum_{i=1}^n (y_i - z_i^{(t)} - [\text{vec}(B)]^\top \text{vec}(\hat{X}_i^\top \hat{A}^{(t)}))^2 + \lambda_B \|\text{vec}(B)\|_2$
      \State $\hat{A}^{(t+1)} \gets \min_A \frac{1}{2n} \sum_{i=1}^n (y_i - z_i^{(t)} - [\text{vec}(A)]^\top \text{vec}(\hat{X}_i^\top \hat{B}^{(t+1)})^2 + \Sigma_{r=1}^{R} \lambda_{A_r} \|\text{vec}(A_r)\|_1$
      \State Orthogonalization: $\hat{A}^{(t+1)} \gets \hat{A}^{(t+1)} (\hat{A}^{(t+1)\top} \hat{A}^{(t+1)})^{-1/2}$
      \State $\hat{C}^{(t+1)} \gets \hat{A}^{(t+1)} \otimes \hat{B}^{(t+1)}$
      \State $\hat{\zeta}^{(t+1)} \gets \min_{\zeta} \frac{1}{2n} \sum_{i=1}^n (y_i - \langle X_i, \hat{C}^{(t+1)} \rangle - Z^\top_i\zeta) + \lambda_{\zeta}\|\zeta\|_2$
      \State $z^{(t+1)} = Z^\top \zeta^{(t+1)}$
    \EndFor
    \State \textbf{return} $\hat{A}^{(T)}$, $\hat{B}^{(T)}$, $\hat{C}^{(T)}$, $\hat{\zeta}^{(T)}$
  \end{algorithmic}
\end{algorithm}

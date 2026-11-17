---
title: "Modes of Convergence of Probability Measures: A Uniform Empirical-Process Case Study"
author: "Rui-Yang Zhang"
date: "November 2026"
abstract: |
  Different notions of convergence of probability measures formalise different ways in which distributions become close. We review weak, Wasserstein-2, total variation, and the two directions of Kullback--Leibler convergence, together with the integral probability metric viewpoint and maximum mean discrepancy. Using i.i.d. observations from the uniform distribution on [0,1], we compare the raw empirical measure with a regularised histogram estimator. The former converges weakly, in Wasserstein-2, and in maximum mean discrepancy, but remains maximally separated in total variation and has infinite KL divergence in both directions. The histogram converges in every considered mode. We conclude by comparing the relative strengths of these modes, paying particular attention to the role of compact support.
---

# Notions of Convergence

Let $P_n$ and $P$ be probability measures on a metric space $(\Theta,d)$, with $P$ the intended limit. Different notions of convergence specify which discrepancies between $P_n$ and $P$ must vanish. Unless otherwise specified, convergence below refers to a deterministic sequence of measures; in the empirical-process example, the resulting assertions will hold almost surely.

## Weak Convergence

We write $P_n\Rightarrow P$ if
$$
\int f\,dP_n\longrightarrow\int f\,dP
$$
for every bounded continuous function $f:\Theta\to\mathbb R$. Weak convergence requires agreement in the expectations of continuous observables, but not necessarily in the probability of every measurable event.

## Total Variation Convergence

The total variation distance is
$$
d_{\mathrm{TV}}(P,Q)=\sup_{A\in\mathcal B(\Theta)}|P(A)-Q(A)|.
$$
When $P$ and $Q$ have densities $p,q$ with respect to a common measure,
$$
d_{\mathrm{TV}}(P,Q)=\frac12\int|p-q|.
$$
Thus $d_{\mathrm{TV}}(P_n,P)\to0$ requires uniform agreement across all measurable event probabilities.

## Forward and Reverse KL Convergence

For $P\ll Q$, define
$$
D_{\mathrm{KL}}(P\|Q)=\int\log\left(\frac{dP}{dQ}\right)dP,
$$
and set the divergence to $+\infty$ otherwise. We distinguish the two directions
$$
D_{\mathrm{KL}}(P_n\|P)\to0
\qquad\text{and}\qquad
D_{\mathrm{KL}}(P\|P_n)\to0.
$$
We call these *forward* and *reverse* KL convergence, respectively, relative to the fixed target $P$ (these labels are not universal). Both control relative density discrepancies, but they are not equivalent in general.

## Wasserstein-2 Convergence

For probability measures with finite second moments,
$$
W_2(P,Q)^2=\inf_{\gamma\in\Gamma(P,Q)}
\int_{\Theta\times\Theta}d(x,y)^2\,d\gamma(x,y),
$$
where $\Gamma(P,Q)$ is the set of couplings with marginals $P$ and $Q$. Wasserstein-2 convergence, $W_2(P_n,P)\to0$, requires that mass can be transported between the distributions at vanishing mean squared cost. On $\mathbb R^d$, it is equivalent to weak convergence together with convergence of second moments.

## Integral Probability Metrics and MMD

An integral probability metric (IPM) associated with a function class $\mathcal F$ is
$$
d_{\mathcal F}(P,Q)=\sup_{f\in\mathcal F}
\left|\int f\,dP-\int f\,dQ\right|.
$$
Total variation is an IPM with $\mathcal F=\{f:0\le f\le1\}$, while Wasserstein-1 corresponds to 1-Lipschitz functions (under the usual moment assumptions). The **maximum mean discrepancy** (MMD) uses the unit ball of a reproducing kernel Hilbert space $\mathcal H_k$:
$$
\operatorname{MMD}_k(P,Q)
=\sup_{\|f\|_{\mathcal H_k}\le1}|P f-Q f|.
$$
In our case study we use the Gaussian kernel $k_\sigma(x,y)=\exp\{-(x-y)^2/(2\sigma^2)\}$ with fixed $\sigma>0$. Its MMD metrises weak convergence on $[0,1]$. KL and Wasserstein-2 are not, in general, IPMs.

# A Uniform Empirical-Process Case Study

Suppose
$$
X_1,X_2,\ldots\overset{\mathrm{iid}}{\sim}P=\operatorname{Unif}[0,1],
\qquad p(x)=1\quad (x\in[0,1]).
$$
The **raw empirical measure** and empirical cumulative distribution function (CDF) are
$$
\widehat P_n=\frac1n\sum_{i=1}^n\delta_{X_i},
\qquad
F_n(x)=\frac1n\sum_{i=1}^n\mathbf1\{X_i\le x\}.
$$
The target CDF is $F(x)=x$ for $x\in[0,1]$. We first analyse $\widehat P_n$ and then introduce a smoothed counterpart $\widetilde P_n$. This distinction is essential: the raw empirical distribution is discrete even when the target is continuous.

## A Smoothed Empirical Measure and a Central Estimate

Let $J_n=\lfloor n^{1/4}\rfloor$ and partition $[0,1]$ into $J_n$ equal bins $B_{1,n},\ldots,B_{J_n,n}$, with endpoint conventions chosen to form a partition. Define the bin counts
$$
N_{j,n}=\sum_{i=1}^n\mathbf1\{X_i\in B_{j,n}\}.
$$
We add one pseudocount per bin and define a **regularised histogram density** by
$$
\boxed{
\widetilde p_n(x)=\frac{J_n(N_{j,n}+1)}{n+J_n},
\qquad x\in B_{j,n}.
}
$$
The corresponding measure is $\widetilde P_n(dx)=\widetilde p_n(x)\,dx$. It is normalised and strictly positive. If $H_n$ denotes the histogram obtained by spreading each empirical atom uniformly over its bin, then
$$
\widetilde P_n=\frac{n}{n+J_n}H_n
+\frac{J_n}{n+J_n}P.
$$
The second term is a vanishing uniform component, which prevents zero-density bins.

Our central estimate is
$$
\boxed{\delta_n:=\|\widetilde p_n-1\|_\infty
=O_{\mathrm{a.s.}}\!\left(n^{-1/4}\sqrt{\log n}\right)\longrightarrow0.}
$$
To verify it, write $\Delta_n=\|F_n-F\|_\infty$. The Dvoretzky--Kiefer--Wolfowitz (DKW) inequality and Borel--Cantelli give
$$
\Delta_n=O_{\mathrm{a.s.}}\!\left(\sqrt{\frac{\log n}{n}}\right).
$$
For every bin, $|N_{j,n}/n-1/J_n|\le2\Delta_n$. Since the uniform population histogram has density exactly one,
$$
\begin{aligned}
\delta_n
&=\max_{1\le j\le J_n}
\left|\frac{J_nN_{j,n}-n}{n+J_n}\right|\\
&\le\frac{2nJ_n}{n+J_n}\Delta_n
\le2J_n\Delta_n.
\end{aligned}
$$
Because $J_n\asymp n^{1/4}$, this tends to zero almost surely. Unlike a non-uniform target, the uniform distribution incurs no histogram approximation bias; only the empirical bin fluctuations remain.

## Weak Convergence

For the raw empirical measure, Glivenko--Cantelli gives
$$
\Delta_n=\sup_{x\in\mathbb R}|F_n(x)-F(x)|
\overset{\mathrm{a.s.}}{\longrightarrow}0.
$$
Uniform CDF convergence implies weak convergence, and therefore
$$
\boxed{\widehat P_n\Rightarrow P\qquad\mathrm{a.s.}}
$$
Equivalently, the strong law gives $n^{-1}\sum_i f(X_i)\to\mathbb E_Pf(X)$ for bounded continuous $f$; the CDF argument supplies a common almost-sure event.

For the smoothed measure, let $f$ be bounded and continuous, and define its modulus of continuity by $\omega_f(h)=\sup_{|x-y|\le h}|f(x)-f(y)|$. Spreading an observation uniformly within its bin changes $f$ by at most $\omega_f(1/J_n)$. From the mixture representation,
$$
\left|\int f\,d\widetilde P_n-\int f\,d\widehat P_n\right|
\le\omega_f(1/J_n)
+\frac{2J_n}{n+J_n}\|f\|_\infty\longrightarrow0.
$$
Together with $\widehat P_n\Rightarrow P$ this yields
$$
\boxed{\widetilde P_n\Rightarrow P\qquad\mathrm{a.s.}}
$$
The proof relies only on the empirical-process law of large numbers and the fact that smoothing becomes negligible for continuous observables.

## Wasserstein-2 Convergence

On $[0,1]$, $|x-y|^2\le|x-y|$; hence $W_2^2(Q,R)\le W_1(Q,R)$. In one dimension,
$$
W_1(\widehat P_n,P)=\int_0^1|F_n(x)-x|\,dx
\le\Delta_n.
$$
It follows that
$$
\boxed{W_2(\widehat P_n,P)\le\sqrt{\Delta_n}
\longrightarrow0\qquad\mathrm{a.s.}}
$$
For $\widetilde P_n$, we construct an explicit coupling with $\widehat P_n$. With probability $n/(n+J_n)$, pair a randomly selected observation with a uniform point in its own bin, whose distance is at most $1/J_n$. Otherwise, pair a randomly selected observation with an independent uniform point in $[0,1]$, whose distance is at most one. Consequently,
$$
W_2^2(\widetilde P_n,\widehat P_n)
\le\frac1{J_n^2}+\frac{J_n}{n+J_n}.
$$
The triangle inequality then gives
$$
\boxed{
W_2(\widetilde P_n,P)
\le\left(\frac1{J_n^2}+\frac{J_n}{n+J_n}\right)^{1/2}
+W_2(\widehat P_n,P)\longrightarrow0
\quad\mathrm{a.s.}}
$$
Here the crucial ingredient is control of the transportation distance, rather than control of density ratios.

## Total Variation Convergence

Let $A_n=\{X_1,\ldots,X_n\}$. For every finite sample, $\widehat P_n(A_n)=1$ whereas $P(A_n)=0$ almost surely. Thus
$$
\boxed{d_{\mathrm{TV}}(\widehat P_n,P)=1
\qquad\text{for every }n\ \mathrm{a.s.}}
$$
In contrast, $\widetilde P_n$ and $P$ have Lebesgue densities, and
$$
\begin{aligned}
d_{\mathrm{TV}}(\widetilde P_n,P)
&=\frac12\int_0^1|\widetilde p_n(x)-1|\,dx\\
&\le\frac12\delta_n\longrightarrow0.
\end{aligned}
$$
Therefore
$$
\boxed{d_{\mathrm{TV}}(\widetilde P_n,P)\longrightarrow0
\qquad\mathrm{a.s.}}
$$
Smoothing removes the discrete-versus-continuous singularity, and uniform density control is more than sufficient for TV convergence.

## Forward KL Convergence

Since $\widehat P_n$ assigns mass to points to which $P$ assigns none, $\widehat P_n\not\ll P$. Consequently,
$$
\boxed{D_{\mathrm{KL}}(\widehat P_n\|P)=+\infty
\qquad\text{for every }n\ \mathrm{a.s.}}
$$
For the histogram, use $D_{\mathrm{KL}}(Q\|R)\le\chi^2(Q\|R)$ and $p\equiv1$:
$$
\begin{aligned}
D_{\mathrm{KL}}(\widetilde P_n\|P)
&\le\int_0^1\frac{(\widetilde p_n(x)-1)^2}{1}\,dx\\
&\le\delta_n^2\longrightarrow0.
\end{aligned}
$$
Hence
$$
\boxed{D_{\mathrm{KL}}(\widetilde P_n\|P)\longrightarrow0
\qquad\mathrm{a.s.}}
$$
The fixed reference density is bounded away from zero, so an absolute density error also controls the relative error entering forward KL.

## Reverse KL Convergence

Because $P$ is nonatomic while $\widehat P_n$ is supported on finitely many points, $P\not\ll\widehat P_n$, giving
$$
\boxed{D_{\mathrm{KL}}(P\|\widehat P_n)=+\infty
\qquad\text{for every }n\ \mathrm{a.s.}}
$$
For the histogram, the pseudocounts ensure $\widetilde p_n>0$ for every $n$. Moreover, $\delta_n\to0$ implies that eventually $\widetilde p_n(x)\ge1/2$ uniformly. Thus
$$
\begin{aligned}
D_{\mathrm{KL}}(P\|\widetilde P_n)
&\le\int_0^1\frac{(1-\widetilde p_n(x))^2}{\widetilde p_n(x)}\,dx\\
&\le2\delta_n^2\longrightarrow0.
\end{aligned}
$$
We conclude that
$$
\boxed{D_{\mathrm{KL}}(P\|\widetilde P_n)\longrightarrow0
\qquad\mathrm{a.s.}}
$$
Unlike forward KL, this bound requires control of the *estimated density in the denominator*. The positive pseudocounts guarantee finite reverse KL at every $n$, and uniform consistency supplies the eventual lower bound.

## Maximum Mean Discrepancy Convergence

Fix a Gaussian kernel $k_\sigma$ and let $\varphi(x)=k_\sigma(x,\cdot)\in\mathcal H_{k_\sigma}$. The empirical MMD is the Hilbert-space norm
$$
\operatorname{MMD}_{k_\sigma}(\widehat P_n,P)
=\left\|\frac1n\sum_{i=1}^n\varphi(X_i)
-\mathbb E_P\varphi(X)\right\|_{\mathcal H_{k_\sigma}}.
$$
Since $\|\varphi(x)\|_{\mathcal H_{k_\sigma}}^2=k_\sigma(x,x)=1$, the Hilbert-space strong law of large numbers gives
$$
\boxed{\operatorname{MMD}_{k_\sigma}(\widehat P_n,P)
\longrightarrow0\qquad\mathrm{a.s.}}
$$
For the smoothed measure, the reproducing property gives $\|f\|_\infty\le1$ whenever $\|f\|_{\mathcal H_{k_\sigma}}\le1$. Hence
$$
\operatorname{MMD}_{k_\sigma}(\widetilde P_n,P)
\le\int_0^1|\widetilde p_n(x)-1|\,dx
\le\delta_n\longrightarrow0.
$$
Thus both measures converge in Gaussian-kernel MMD, even though the raw empirical measure fails to converge in TV or either KL direction.

# Relative Strengths of the Convergences

## General Implications

For probability measures on a Polish space, Pinsker's inequality gives, in either direction,
$$
\begin{aligned}
d_{\mathrm{TV}}(P_n,P)
&\le\sqrt{\tfrac12D_{\mathrm{KL}}(P_n\|P)},\\
d_{\mathrm{TV}}(P_n,P)
&\le\sqrt{\tfrac12D_{\mathrm{KL}}(P\|P_n)}.
\end{aligned}
$$
Therefore **either KL convergence implies TV convergence**, which in turn implies weak convergence. Also, $W_2$ convergence implies weak convergence whenever the $W_2$ distances are defined. Neither KL direction implies the other in general.

On an unbounded space, TV (or even KL in both directions) need not imply $W_2$: a vanishing amount of mass can move arbitrarily far away. Conversely, $W_2$ convergence need not imply TV convergence, as the discrete-versus-continuous example already shows. For a bounded Gaussian kernel, TV convergence implies MMD convergence; a Gaussian kernel on $\mathbb R^d$ also metrises weak convergence.

## The Simplified Hierarchy on $[0,1]$

Compact support makes the relationship particularly clean. Since $d(x,y)\le1$,
$$
W_2(P_n,P)^2\le d_{\mathrm{TV}}(P_n,P).
$$
Furthermore, on a compact metric space, $W_2$ convergence and weak convergence are equivalent. For the fixed Gaussian kernel considered here, MMD convergence is also equivalent to weak convergence. Thus the hierarchy on $[0,1]$ is
$$
\boxed{
\begin{gathered}
D_{\mathrm{KL}}(P_n\|P)\to0
\quad\text{or}\quad
D_{\mathrm{KL}}(P\|P_n)\to0
\\[3pt]
\Downarrow\\[-1pt]
d_{\mathrm{TV}}(P_n,P)\to0
\\[-1pt]\Downarrow\\[-1pt]
W_2(P_n,P)\to0
\quad\Longleftrightarrow\quad
P_n\Rightarrow P
\quad\Longleftrightarrow\quad
\operatorname{MMD}_{k_\sigma}(P_n,P)\to0.
\end{gathered}}
$$
The implication from TV to $W_2$ is strict: the raw empirical measure converges in $W_2$ but stays at TV distance one. The implication from either KL direction to TV is also strict; density sequences may converge in $L^1$ while having extreme log-density ratios on small sets. In fact, the two KL directions are incomparable even on $[0,1]$. Relative to the uniform target, a density that equals $e^{-n^2}$ on an interval of length $1/n$ (and is renormalised elsewhere) has forward KL tending to zero but reverse KL diverging. Conversely, assigning mass $1/n$ to an interval of length $e^{-n^2}$ (and distributing the remaining mass uniformly over its complement) gives reverse KL tending to zero but forward KL diverging.

## Conclusions from the Case Study

| Mode | Raw empirical measure $\widehat P_n$ | Smoothed histogram $\widetilde P_n$ |
|:--|:--:|:--:|
| Weak | Converges | Converges |
| Wasserstein-2 | Converges | Converges |
| Total variation | Distance always $1$ | Converges |
| Forward KL | Always $+\infty$ | Converges |
| Reverse KL | Always $+\infty$ | Converges |
| Gaussian MMD | Converges | Converges |

The difference between the two estimators is not the amount of data but their **measure-theoretic structure**. Empirical-process convergence suffices for weak and Wasserstein convergence on a compact domain, and bounded-kernel MMD is equally permissive. To obtain TV and KL convergence to a continuous target, we must first avoid the singularity of a purely atomic empirical measure. Uniform consistency of the positive histogram density then provides a convenient common argument for all the stronger modes. These conditions are sufficient rather than necessary; in particular, the equivalence between weak and Wasserstein convergence depends crucially on compact support.

## References

- Billingsley, P. (1999). *Convergence of Probability Measures*, 2nd ed. Wiley.
- van der Vaart, A. W. (1998). *Asymptotic Statistics*. Cambridge University Press.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., and Smola, A. (2012). A kernel two-sample test. *Journal of Machine Learning Research*, 13, 723--773.

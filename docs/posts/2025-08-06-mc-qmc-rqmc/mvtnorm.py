#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Define the integrand
def f(x):
    return x**2

# True value of the integral
true_value = 1/12

# Basic Monte Carlo Estimator
def mc_estimate(n):
    x = np.random.uniform(0, 1, n)
    return np.mean(f(x))

# Antithetic Variates Estimator
def antithetic_estimate(n):
    u = np.random.uniform(0, 0.5, n)
    x = np.concatenate([u, 1 - u])
    return np.mean(f(x))

# Quasi-Monte Carlo using Sobol sequence
def qmc_estimate(n):
    sampler = qmc.Sobol(d=1, scramble=False)
    x = sampler.random(n)
    return np.mean(f(x))

# Randomized QMC (scrambled Sobol)
def rqmc_estimate(n):
    sampler = qmc.Sobol(d=1, scramble=True)
    x = sampler.random(n)
    return np.mean(f(x))

# Run multiple trials and compute MSE
def run_mse(estimator, n, trials=10000):
    estimates = np.array([estimator(n) for _ in range(trials)])
    mse = np.mean((estimates - true_value)**2)
    return mse

# Compare MSEs for different sample sizes
sample_sizes = [2**i for i in range(6, 12)]
mse_mc = [run_mse(mc_estimate, n) for n in sample_sizes]
mse_antithetic = [run_mse(antithetic_estimate, n//2) for n in sample_sizes]  # each half = n/2
mse_qmc = [run_mse(qmc_estimate, n) for n in sample_sizes]
mse_rqmc = [run_mse(rqmc_estimate, n) for n in sample_sizes]

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(sample_sizes, mse_mc, label='MC', marker='o')
plt.loglog(sample_sizes, mse_antithetic, label='Antithetic', marker='s')
plt.loglog(sample_sizes, mse_qmc, label='QMC (Sobol)', marker='^')
plt.loglog(sample_sizes, mse_rqmc, label='Randomized QMC', marker='x')
plt.xlabel('Sample size (log scale)')
plt.ylabel('MSE (log scale)')
plt.title('Average MSE for Estimating $E[||X||^2]$ for Unif[0,1]')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm

# -----------------------------
# Configuration
# -----------------------------
d = 4                          # Dimension
trials = 500                    # Number of trials for MSE estimation
sample_sizes = [2**i for i in range(4, 12)]
mu = np.zeros(d)                # Mean vector (zero)

# -----------------------------
# Function: Integrand
# -----------------------------
def f_norm2(x):
    """Compute squared L2 norm for each row vector."""
    return np.sum(x**2, axis=1)

# -----------------------------
# Box-Muller Sampling
# -----------------------------
def box_muller(u1, u2, eps=1e-10):
    """Convert uniform samples to standard normals using Box-Muller."""
    u1 = np.clip(u1, eps, 1.0)  # Prevent log(0)
    r = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)
    return z1, z2

def generate_standard_normals(n_samples, d, sampler=None):
    """Generate standard normal samples via Box-Muller."""
    assert d % 2 == 0, "Box-Muller requires even dimension"
    total = n_samples * d // 2
    u = sampler.random(total) if sampler else np.random.uniform(0, 1, size=(total, 2))
    z1, z2 = box_muller(u[:, 0], u[:, 1])
    z = np.stack([z1, z2], axis=1).reshape(n_samples, d)
    return z

# -----------------------------
# Correlated Multivariate Normal
# -----------------------------
def generate_random_correlation_matrix(d, seed=None):
    """Generate a random correlation matrix with diagonal 1s."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.uniform(-1, 1, size=(d, d))
    Sigma = A @ A.T
    stddev = np.sqrt(np.diag(Sigma))
    return Sigma / stddev[:, None] / stddev[None, :]


Sigma = generate_random_correlation_matrix(d, seed=42)#np.eye(d)

def transform_to_mvnormal(z, mu, Sigma):
    """Transform standard normals to multivariate normal with given mu and Sigma."""
    L = np.linalg.cholesky(Sigma)
    return mu + z @ L.T

# -----------------------------
# Estimators
# -----------------------------
def mc_estimate(n, d):
    z = generate_standard_normals(n, d)
    x = transform_to_mvnormal(z, mu, Sigma)
    return np.mean(f_norm2(x))

def antithetic_estimate(n, d):
    z = generate_standard_normals(n, d)
    z_anti = -z
    x = transform_to_mvnormal(z, mu, Sigma)
    x_anti = transform_to_mvnormal(z_anti, mu, Sigma)
    return np.mean((f_norm2(x) + f_norm2(x_anti)) / 2)

def qmc_estimate(n, d):
    assert d % 2 == 0, "Dimension must be even"
    sampler = qmc.Sobol(d=2, scramble=False)
    total_pairs = n * d // 2
    u = sampler.random(total_pairs)
    z1, z2 = box_muller(u[:, 0], u[:, 1])
    z = np.concatenate([z1, z2])
    z = z.reshape(n, d)
    x = transform_to_mvnormal(z, mu, Sigma)
    return np.mean(f_norm2(x))

def rqmc_estimate(n, d):
    assert d % 2 == 0, "Dimension must be even"
    sampler = qmc.Sobol(d=2, scramble=True)
    total_pairs = n * d // 2
    u = sampler.random(total_pairs)
    z1, z2 = box_muller(u[:, 0], u[:, 1])
    z = np.concatenate([z1, z2])
    z = z.reshape(n, d)
    x = transform_to_mvnormal(z, mu, Sigma)
    return np.mean(f_norm2(x))

# -----------------------------
# Run trials and compute MSE
# -----------------------------
def average_mse(estimator, n, d, trials=100):
    estimates = [estimator(n, d) for _ in range(trials)]
    return np.mean((np.array(estimates) - d)**2)

# -----------------------------
# Run experiments
# -----------------------------
mse_mc =           [average_mse(mc_estimate,        n,     d, trials) for n in sample_sizes]
mse_antithetic =   [average_mse(antithetic_estimate, n//2, d, trials) for n in sample_sizes]
mse_qmc =          [average_mse(qmc_estimate,       n,     d, trials=1) for n in sample_sizes]  # deterministic
mse_rqmc =         [average_mse(rqmc_estimate,      n,     d, trials) for n in sample_sizes]

# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(10, 6))
plt.loglog(sample_sizes, mse_mc, label='MC', marker='o')
plt.loglog(sample_sizes, mse_antithetic, label='Antithetic', marker='s')
plt.loglog(sample_sizes, mse_qmc, label='QMC (Sobol)', marker='^')
plt.loglog(sample_sizes, mse_rqmc, label='Randomized QMC', marker='x')

# Y-axis bounds
all_mse = np.concatenate([mse_mc, mse_antithetic, mse_qmc, mse_rqmc])
finite_mse = all_mse[np.isfinite(all_mse)]
if finite_mse.size > 0:
    ymin, ymax = finite_mse.min() * 0.8, finite_mse.max() * 1.2
    plt.ylim(ymin, ymax)

plt.xlabel('Sample size (log scale)')
plt.ylabel('Average MSE (log scale)')
plt.title(f'Average MSE for Estimating $E[||X||^2]$ with MVN($\\mu$, $\\Sigma$), $d={d}$')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.rcParams.update({'font.size': 10})
#plt.savefig(f"mvtnorm_diag_mse_d{d}.png",dpi=300)
plt.show()


#%%
n = 2**10
samples = generate_standard_normals(n, d)
X = transform_to_mvnormal(samples, mu, Sigma)
X_anti = transform_to_mvnormal(-samples, mu, Sigma)

f_vals = f_norm2(X)
f_vals_anti = f_norm2(X_anti)

corr = np.corrcoef(f_vals, f_vals_anti)[0, 1]
print("Correlation between f(x) and f(-x):", corr)

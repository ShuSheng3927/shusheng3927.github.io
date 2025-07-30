#%% Imports
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns
import blackjax

sns.set_theme(style="whitegrid")

#%% Define ODE and true parameters
# ODE: dx/dt = exp(-l*t) * cos(a*t)
def drift(x, t, a, l):
    return jnp.exp(-l * t) * jnp.cos(a * t)

# Simulation settings
start_time, term_time, dt = 0.0, 5.0, 0.01
tt = jnp.arange(start_time, term_time + dt, dt)

# Ground truth parameters
true_params = {"x0": 1.0, "a": 5.0, "l": 0.5}
x0_true, a_true, l_true = true_params.values()

# Simulate latent dynamics using Euler method
x_curr = x0_true
x_traj = [x_curr]
for t in tt[1:]:
    x_curr += dt * drift(x_curr, t, a_true, l_true)
    x_traj.append(x_curr)
x_traj = jnp.array(x_traj)

# Plot true trajectory
plt.plot(tt, x_traj)
plt.title(r"$\frac{dx}{dt} = \exp(-lt)\cos(at)$")
plt.xlabel("Time $t$")
plt.ylabel("State $x$")
plt.grid(True)
plt.show()

#%% Generate noisy observations
key = jr.key(1)
n_obs = 100
noise = 0.05

obs_idx = jnp.sort(jr.choice(key, tt.shape[0], (n_obs,), replace=False))
obs_time = tt[obs_idx]
obs_val = x_traj[obs_idx] + jr.normal(key, shape=obs_time.shape) * noise
obs_data = jnp.stack([obs_time, obs_val], axis=1)

# Plot observations with latent dynamics
plt.plot(tt, x_traj, label="Latent Dynamics")
plt.plot(obs_time, obs_val, "rx", label="Observations")
plt.xlabel("Time $t$")
plt.ylabel("State $x$")
plt.legend()
plt.grid(True)
plt.show()

#%% Log-likelihood function
@jax.jit
def loglikelihood(theta):
    x0 = theta["x0"]
    obs_noise = theta["obs_noise"]
    a = theta["a"]
    l = theta["l"]

    x_curr = x0
    residuals = []

    def compute_residual(t, x_val):
        mask = obs_data[:, 0] == t  # Boolean mask for time t
        obs = obs_data[:, 1]
        res = jnp.where(mask, obs - x_val, 0.0)
        return res

    residuals.append(compute_residual(0, x_curr))

    for t in tt[1:]:
        x_curr = x_curr + dt * drift(x_curr, t, a, l)
        residuals.append(compute_residual(t, x_curr))

    all_residuals = jnp.stack(residuals)
    all_residuals = all_residuals.reshape(-1)
    time_mask = jnp.isin(obs_data[:, 0], tt)
    full_mask = jnp.tile(time_mask, (len(tt),))  # Expand to match residuals
    filtered_residuals = jnp.where(full_mask, all_residuals, jnp.nan)
    valid_residuals = jnp.nan_to_num(filtered_residuals, nan=0.0)

    loglik = jnp.sum(jax.scipy.stats.norm.logpdf(valid_residuals, scale=obs_noise))
    return loglik

#%% MCMC setup
def build_rw_kernel(sigmas):
    proposal = blackjax.mcmc.random_walk.normal(sigmas)
    return blackjax.additive_step_random_walk(loglikelihood, proposal)

def step(key, state, sigmas):
    kernel = build_rw_kernel(sigmas)
    return kernel.step(key, state)

#%% Adaptive tuning (Robbins–Monro)
def calibration_loop(rng_key, init_state, init_sigmas, num_steps,
                     alpha_star=0.4, gamma0=0.3, kappa=0.7):
    log_sig = jnp.log(init_sigmas)

    def body(carry, i):
        state, log_sig, key = carry
        key, subkey = jax.random.split(key)
        sigmas = jnp.exp(log_sig)
        state, info = step(subkey, state, sigmas)
        gamma = gamma0 / ((i + 1.0) ** kappa)
        log_sig += gamma * (info.is_accepted - alpha_star)
        return (state, log_sig, key), state

    carry0 = (init_state, log_sig, rng_key)
    (state, log_sig, _), burnin_states = jax.lax.scan(body, carry0, jnp.arange(num_steps))
    return state, jnp.exp(log_sig), burnin_states

#%% Main MCMC loop
def inference_loop(rng_key, init_sigmas, num_samples, num_burnin):
    init_state = build_rw_kernel(init_sigmas).init(initial_position)
    rng_key, subkey = jax.random.split(rng_key)

    # Burn-in with adaptation
    state, tuned_sigmas, burnin_states = calibration_loop(subkey, init_state,
                                                          init_sigmas, num_burnin)

    # Sampling
    @jax.jit
    def one_step(carry, key):
        state, acc_sum, i = carry
        state, info = step(key, state, tuned_sigmas)
        acc_sum += info.is_accepted
        return (state, acc_sum, i + 1), state

    keys = jax.random.split(rng_key, num_samples)
    (final_state, acc_sum, iters), sample_states = jax.lax.scan(
        one_step, (state, 0.0, 0), keys)

    acc_rate = acc_sum / iters
    return burnin_states.position, sample_states.position, tuned_sigmas, acc_rate
#%% Run the full MCMC procedure
initial_position = {"a": 4.0, "l": 1.0, "x0": 0.5}

rng_key = jr.key(0)
init_sigmas = jnp.array([0.1, 0.1, 0.01])
burnin_states, states, tuned_sigmas, acc_rate = inference_loop(
    rng_key, init_sigmas, num_samples=20_000, num_burnin=20_000
)

full_states = {k: jnp.concatenate([burnin_states[k], states[k]]) for k in states}

print("Tuned step sizes (σ):", tuned_sigmas)
print("Acceptance rate:", float(acc_rate))

# MCMC full trace plots
fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
for i, k in enumerate(initial_position):
    axes[i].plot(full_states[k])
    axes[i].axhline(true_params[k], ls="--", color="red")
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel(k)
plt.tight_layout()
plt.show()

# Burn-in Removed Trace plots and histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
row_labels = ["Trace", "Histogram"]

for i, k in enumerate(initial_position):
    # Trace plot
    axes[0, i].plot(states[k], color="steelblue")
    axes[0, i].axhline(true_params[k], ls="--", color="red", label="True")
    axes[0, i].set_title(f"MCMC Trace: {k}")
    axes[0, i].legend()

    # Histogram
    axes[1, i].hist(states[k], bins=30, density=True, color="lightseagreen", alpha=0.7)
    axes[1, i].axvline(true_params[k], ls="--", color="red", label="True")
    axes[1, i].legend()

plt.tight_layout()
plt.show()

#%% Posterior predictive trajectory
@jax.jit
def simulate_traj(x0, a, l):
    x_curr = x0
    traj = [x_curr]
    for t in tt[1:]:
        x_curr += dt * drift(x_curr, t, a, l)
        traj.append(x_curr)
    return jnp.array(traj)

# Bootstrap posterior samples
num_samples = 1000
alpha = 0.1  # for 90% CI
posterior_trajs = []

for i in range(num_samples):
    idx = jr.choice(jr.key(i), states["a"].shape[0], (1,))
    a_sample, l_sample, x0_sample = states["a"][idx][0], states["l"][idx][0], states["x0"][idx][0]
    posterior_trajs.append(simulate_traj(x0_sample, a_sample, l_sample))

posterior_trajs = jnp.array(posterior_trajs)
mean_traj = jnp.mean(posterior_trajs, axis=0)
lower = jnp.percentile(posterior_trajs, alpha / 2 * 100, axis=0)
upper = jnp.percentile(posterior_trajs, (1 - alpha / 2) * 100, axis=0)

# Plot predictive mean and CI
plt.fill_between(tt, lower, upper, color='green', alpha=0.2, label=f"{int((1-alpha)*100)}% CI")
plt.plot(tt, mean_traj, color='green', label="Posterior Mean", linewidth=2)
plt.plot(tt, x_traj, label="True Dynamics", linewidth=1.5)
plt.plot(obs_time, obs_val, "rx", label="Observations")
plt.xlabel("Time $t$")
plt.ylabel("State $x$")
plt.legend()
plt.grid(True)
plt.show()


#%% NUTS
import numpy as np 

inv_mass_matrix = np.array([1,1,1])
step_size = 1e-3
nuts = blackjax.nuts(loglikelihood, step_size, inv_mass_matrix)

initial_position = {"a": 4., "l": 0.5, "x0": 2.}
initial_state = nuts.init(initial_position)

warmup = blackjax.window_adaptation(blackjax.nuts, loglikelihood)
rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
(state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=1_000)

kernel = blackjax.nuts(loglikelihood, **parameters).step
states = inference_loop(sample_key, kernel, state, 1_000)

mcmc_samples = states.position

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
row_labels = ["Trace", "Histogram"]

for i, k in enumerate(initial_position):
    # Trace plot
    axes[0, i].plot(mcmc_samples[k], color="steelblue")
    axes[0, i].axhline(true_params[k], ls="--", color="red", label="True")
    axes[0, i].set_title(f"MCMC Trace: {k}")
    axes[0, i].legend()

    # Histogram
    axes[1, i].hist(mcmc_samples[k], bins=30, density=True, color="lightseagreen", alpha=0.7)
    axes[1, i].axvline(true_params[k], ls="--", color="red", label="True")
    axes[1, i].legend()

plt.tight_layout()
plt.show()
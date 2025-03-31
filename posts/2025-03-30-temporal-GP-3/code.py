#%%
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import expm
import matplotlib.pyplot as plt
import time
jax.config.update("jax_enable_x64", True)

#%%
def kalman_filter(time_grid, y, f0, P0, obs_idx, Phi, Q, H, sig):
    """
    Run the Kalman filter on a fine time grid (e.g. dt = 0.01) while performing 
    measurement updates only when an observation is available.
    
    Parameters:
        time_grid: array of time points (e.g., every 0.01 sec)
        y       : array of observations corresponding to obs_idx times
        f0, P0  : initial state estimate and covariance
        obs_idx : indices in time_grid where measurements are available
        Phi     : state transition matrix computed for dt (e.g., expm(F*dt))
        Q       : process noise covariance computed for dt
        H       : measurement matrix
        sig     : measurement noise standard deviation
    
    Returns:
        f_filt : filtered state estimates for every time step in time_grid (shape: [N, state_dim])
        P_filt : filtered covariances (shape: [N, state_dim, state_dim])
        f_pred : predicted state estimates for steps (shape: [N-1, state_dim])
        P_pred : predicted covariances (shape: [N-1, state_dim, state_dim])
    """
    N = time_grid.shape[0]
    state_dim = f0.shape[0]
    
    # Initialize lists to store estimates at each time step
    f_filt = [f0]
    P_filt = [P0]
    f_pred = []
    P_pred = []
    
    # Create a set for quick lookup of observation indices and a counter for y
    obs_set = set(obs_idx.tolist())
    obs_counter = 0

    for k in range(N - 1):
        # Prediction step for one dt step
        f_p = Phi @ f_filt[-1]
        P_p = Phi @ P_filt[-1] @ Phi.T + Q
        f_pred.append(f_p)
        P_pred.append(P_p)
        
        # If the next time index corresponds to an observation, perform measurement update
        if (k + 1) in obs_set:
            # Get the observation corresponding to this time point
            y_obs = y[obs_counter]
            obs_counter += 1

            # Compute innovation covariance and Kalman gain
            S = (H @ P_p @ H.T)[0,0] + sig**2  # scalar innovation covariance
            K = P_p @ H.T / S          # gain; division is valid because S is scalar

            # Innovation (residual)
            innovation = y_obs - (H @ f_p)[0]

            # Update state and covariance
            f_new = f_p + K.flatten() * innovation  # use flatten if K is (state_dim, 1)
            P_new = P_p - K @ H @ P_p
        else:
            # No observation: predicted state becomes the filtered estimate
            f_new = f_p
            P_new = P_p
        
        f_filt.append(f_new)
        P_filt.append(P_new)
    
    return jnp.stack(f_filt), jnp.stack(P_filt), jnp.stack(f_pred), jnp.stack(P_pred)

def rts_smoother(f_filt, P_filt, f_pred, P_pred, Phi):
    """
    Rauch–Tung–Striebel (RTS) smoother for fine-grid filtering estimates.
    
    Parameters:
        f_filt : filtered state estimates at every time step (shape: [N, state_dim])
        P_filt : filtered covariances (shape: [N, state_dim, state_dim])
        f_pred : predicted state estimates (shape: [N-1, state_dim])
        P_pred : predicted covariances (shape: [N-1, state_dim, state_dim])
        Phi    : state transition matrix (for dt steps)
        
    Returns:
        f_smooth : smoothed state estimates (shape: [N, state_dim])
        P_smooth : smoothed state covariances (shape: [N, state_dim, state_dim])
    """
    N = f_filt.shape[0]
    state_dim = f_filt.shape[1]
    
    # Initialize lists for smoothed estimates
    f_smooth = [f_filt[-1]]
    P_smooth = [P_filt[-1]]
    
    # Backward recursion: from time N-2 down to 0
    for k in range(N - 2, -1, -1):
        # Compute the smoothing gain: C_k = P_filt[k] * Phi.T * inv(P_pred[k])
        Ck = P_filt[k] @ Phi.T @ jax.numpy.linalg.inv(P_pred[k])
        
        # Smoothed state and covariance at time k
        f_k_smooth = f_filt[k] + Ck @ (f_smooth[0] - f_pred[k])
        P_k_smooth = P_filt[k] + Ck @ (P_smooth[0] - P_pred[k]) @ Ck.T
        
        # Insert at the beginning of the lists
        f_smooth.insert(0, f_k_smooth)
        P_smooth.insert(0, P_k_smooth)
    
    return jnp.stack(f_smooth), jnp.stack(P_smooth)

def kalman_likelihood(y, f0, P0, Phi, Q, H, sig):
    """
    Compute the log marginal likelihood of the observation sequence using the Kalman filter.
    
    Parameters:
        y   : array of observations (shape: [N])
        f0  : initial state estimate (shape: [state_dim])
        P0  : initial state covariance (shape: [state_dim, state_dim])
        Phi : state transition matrix for the observation interval (e.g., expm(F*delta))
        Q   : process noise covariance for the observation interval
        H   : measurement matrix
        sig : measurement noise standard deviation

    Returns:
        loglik : the accumulated log marginal likelihood (a scalar)
    """
    N = y.shape[0]
    loglik = 0.0
    f_curr = f0
    P_curr = P0

    for k in range(N):
        # Prediction step: propagate state to next observation time
        f_pred = Phi @ f_curr
        P_pred = Phi @ P_curr @ Phi.T + Q

        # Compute innovation (residual) and its covariance (extract scalar)
        S = (H @ P_pred @ H.T)[0, 0] + sig**2
        innovation = y[k] - (H @ f_pred)[0]

        # Accumulate log likelihood from the Gaussian innovation density
        loglik += -0.5 * (jnp.log(2 * jnp.pi) + jnp.log(S) + (innovation**2) / S)

        # Measurement update:
        K = P_pred @ H.T / S  # Kalman gain
        f_curr = f_pred + K.flatten() * innovation
        P_curr = P_pred - K @ H @ P_pred

    return loglik

def compute_neg_log_marginal_likelihood(params, y):
    """
    Given the parameter vector params = [obs_noise, l, sigma2], build the 
    state-space representation for the Matérn 3/2 kernel, run the Kalman filter,
    and return the negative log marginal likelihood of the observations.
    """
    obs_time = y[0,]
    obs_val = y[1,]
    
    obs_noise, l, sigma2 = params
    # Define kernel parameters and corresponding state-space matrices:
    lambda_val = jnp.sqrt(3.0) / l
    F = jnp.array([[0.0, 1.0],
                   [-lambda_val**2, -2.0 * lambda_val]])
    H = jnp.array([[1.0, 0.0]])
    # Steady-state covariance:
    P_inf = jnp.diag(jnp.array([sigma2, lambda_val**2 * sigma2]))
    
    # Use the fine-grid time step (dt)
    dt = jnp.diff(obs_time)[0]  #assuming regular observation times
    Phi = expm(F * dt)
    Q = P_inf - Phi @ P_inf @ Phi.T

    # Use the steady-state as the initial condition
    f0 = jnp.zeros(2)
    P0 = P_inf

    # Run the Kalman filter (which accumulates the log likelihood)
    loglik = kalman_likelihood(obs_val, f0, P0, Phi, Q, H, obs_noise)
    return -loglik


#%%
# simulate ground truth

# Parameters for the Matérn 3/2 kernel
sigma2 = 1.0         # process variance
l = 1.0              # length-scale
lambda_val = jnp.sqrt(3.0) / l

# Define the state-space matrices (companion form)
F = jnp.array([[0.0, 1.0],
               [-lambda_val**2, -2.0 * lambda_val]])
L = jnp.array([[0.0],
               [1.0]])
H = jnp.array([[1.0, 0.0]])

# Simulation parameters
T2 = 6.0     # total time
T = 5.0           
dt = 0.01          # time step
time_grid = jnp.arange(0, T2 + dt, dt)
n_steps = time_grid.shape[0]

Phi = expm(F * dt)
P_inf = jnp.diag(jnp.array([sigma2, lambda_val**2 * sigma2]))
Q = P_inf - Phi @ P_inf @ Phi.T

# Initialize the state at stationarity: f(0) ~ N(0, P_inf)
key = jr.PRNGKey(0)
key, subkey = jr.split(key)
f0 = jr.multivariate_normal(subkey, mean=jnp.zeros(2), cov=P_inf)
f_list = [f0]
y_list = [H @ f0]

# Simulate the state-space model: f_{k+1} = Phi * f_k + e_k, e_k ~ N(0, Q)
current_f = f0
for _ in range(n_steps - 1):
    key, subkey = jr.split(key)
    ek = jr.multivariate_normal(subkey, mean=jnp.zeros(2), cov=Q)
    current_f = Phi @ current_f + ek
    f_list.append(current_f)
    y_list.append(H @ current_f)

# Convert lists to arrays
f = jnp.stack(f_list)
y = jnp.stack(y_list).flatten()

# Plot the generated sample path from the Matérn 3/2 GP
plt.figure(figsize=(8, 4))
plt.plot(time_grid, y, lw=2, color='blue')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Exact Sample from Matérn 3/2 GP')
plt.show()


#%%
# generate observations

delta = 0.1
obs_noise = 0.1
obs_time = jnp.linspace(delta, T, int(T/delta ))
obs_idx = jnp.array([jnp.where(jnp.isclose(time_grid, t))[0][0] for t in obs_time])
obs_val = y[obs_idx] + jr.normal(key, obs_time.shape) * obs_noise

plt.figure(figsize=(8, 4))
plt.plot(time_grid, y, lw=2, color='blue')
plt.plot(obs_time, obs_val,"rx")
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Exact Sample from Matérn 3/2 GP with Observations')
plt.show()


#%%
# regression via SDE approach

f_filt, P_filt, f_pred, P_pred = kalman_filter(time_grid,obs_val, f0, P_inf, obs_idx, Phi, Q, H, obs_noise)
f_smooth, P_smooth = rts_smoother(f_filt, P_filt, f_pred, P_pred, Phi)

plt.figure(figsize=(7.5, 2.5))
plt.plot(time_grid,y, label='Truth', linewidth=2)
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.scatter(obs_time, obs_val, label='Observations',color="red", s=10, alpha=0.6)

filt_means = f_filt[:, 0]
filt_vars = jnp.array([P[0, 0] for P in P_filt])
filt_upper = filt_means + 2 * jnp.sqrt(filt_vars)
filt_lower = filt_means - 2 * jnp.sqrt(filt_vars)
plt.plot(time_grid, filt_means, label='Filtered Mean',color="orange", linestyle='-.', linewidth=1)
plt.fill_between(time_grid, filt_lower, filt_upper, color='orange', alpha=0.3, label='Filtered ±2σ')

smooth_means = f_smooth[:, 0]
smooth_vars = jnp.array([P[0, 0] for P in P_smooth])
smooth_upper = smooth_means + 2 * jnp.sqrt(smooth_vars)
smooth_lower = smooth_means - 2 * jnp.sqrt(smooth_vars)
plt.plot(time_grid, smooth_means, label='Smoothed Mean', color="green")
plt.fill_between(time_grid, smooth_lower, smooth_upper, color='green', alpha=0.2, label='Smoothed ±2σ')

plt.title("GP Regression via Kalman Smoothing")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


#%%
# estimate prior hyperparameter using MLE, then regression

from scipy.optimize import minimize

full_obs = jnp.stack([obs_time, obs_val])
def objective(params):
    # Ensure params is a JAX array
    params = jnp.array(params)
    return compute_neg_log_marginal_likelihood(params, full_obs)
grad_objective = jax.grad(objective)

initial_guess = [1, 1, 2]
result = minimize(objective, initial_guess, method='L-BFGS-B', jac=grad_objective)

optim_param = jnp.array(result.x)
obs_noise_est, l_est, sigma2_est = optim_param
lambda_val_est = jnp.sqrt(3.0) / l_est

# Define the state-space matrices (companion form)
F_est = jnp.array([[0.0, 1.0],
               [-lambda_val_est**2, -2.0 * lambda_val_est]])
L = jnp.array([[0.0],
               [1.0]])
H = jnp.array([[1.0, 0.0]])
P_inf_est = jnp.diag(jnp.array([sigma2_est, lambda_val_est**2 * sigma2_est]))

# Simulation parameters
T2 = 6.0     # total time
T = 5.0           
dt = 0.01          # time step
time_grid = jnp.arange(0, T2 + dt, dt)
n_steps = time_grid.shape[0]

Phi_est = expm(F_est * dt)
Q_est = P_inf_est - Phi_est @ P_inf_est @ Phi_est.T

f_filt, P_filt, f_pred, P_pred = kalman_filter(time_grid,obs_val, f0, P_inf, obs_idx, Phi, Q, H, obs_noise)
f_smooth, P_smooth = rts_smoother(f_filt, P_filt, f_pred, P_pred, Phi)

plt.figure(figsize=(7.5, 2.5))
plt.plot(time_grid,y, label='Truth', linewidth=2)
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.scatter(obs_time, obs_val, label='Observations',color="red", s=10, alpha=0.6)

filt_means = f_filt[:, 0]
filt_vars = jnp.array([P[0, 0] for P in P_filt])
filt_upper = filt_means + 2 * jnp.sqrt(filt_vars)
filt_lower = filt_means - 2 * jnp.sqrt(filt_vars)
plt.plot(time_grid, filt_means, label='Filtered Mean',color="orange", linestyle='-.', linewidth=1)
plt.fill_between(time_grid, filt_lower, filt_upper, color='orange', alpha=0.3, label='Filtered ±2σ')

smooth_means = f_smooth[:, 0]
smooth_vars = jnp.array([P[0, 0] for P in P_smooth])
smooth_upper = smooth_means + 2 * jnp.sqrt(smooth_vars)
smooth_lower = smooth_means - 2 * jnp.sqrt(smooth_vars)
plt.plot(time_grid, smooth_means, label='Smoothed Mean', color="green")
plt.fill_between(time_grid, smooth_lower, smooth_upper, color='green', alpha=0.2, label='Smoothed ±2σ')

plt.title("GP Regression via Kalman Smoothing with Optimised Hyperparameters")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


#%%
# regression comparison: conjugacy v.s. kalman

xtest = time_grid.reshape(-1,1)
ytest = y.reshape(-1,1)

xobs = obs_time.reshape(-1,1)
yobs = obs_val.reshape(-1,1)
D = gpx.Dataset(X = xobs, y=yobs)

kernel = gpx.kernels.Matern32(lengthscale=l, variance=sigma2)  # 1-dimensional input
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev = obs_noise)
posterior = prior * likelihood

latent_dist = posterior.predict(xtest, train_data=D)
predictive_dist = posterior.likelihood(latent_dist)
predictive_mean = predictive_dist.mean()
predictive_std = predictive_dist.stddev()

# fig, ax = plt.subplots(figsize=(15, 2.5))
# ax.plot(
#     xtest, ytest, label="Truth", linewidth=2
# )
# ax.scatter(obs_time, obs_val, label="Observations",color="red", s=10, alpha=0.6)
# ax.plot(xtest, predictive_mean, label="Predictive mean", color="green")
# ax.fill_between(
#     xtest.squeeze(),
#     predictive_mean - 2 * predictive_std,
#     predictive_mean + 2 * predictive_std,
#     alpha=0.2,
#     label="Predictive ±2σ",
#     color="green",
# )
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# ax.set_title("GP Regression via Conjugacy")

fig,ax = plt.subplots(2,1,figsize=(15,5))

ax[0].plot(time_grid, y, label='Truth', linewidth=2)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('y(t)')
ax[0].scatter(obs_time, obs_val, label='Observations',color="red", s=10, alpha=0.6)
ax[0].plot(time_grid, smooth_means, label='Smoothed Mean', color="green")
ax[0].fill_between(time_grid, smooth_lower, smooth_upper, color='green', alpha=0.2, label='Smoothed ±2σ')
ax[0].set_title("GP Regression via Kalman Smoothing")
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

ax[1].plot(
    xtest, ytest, label="Truth", linewidth=2
)
ax[1].set_xlabel('Time')
ax[1].set_ylabel('y(t)')
ax[1].scatter(obs_time, obs_val, label="Observations",color="red", s=10, alpha=0.6)
ax[1].plot(xtest, predictive_mean, label="Predictive mean", color="green")
ax[1].fill_between(
    xtest.squeeze(),
    predictive_mean - 2 * predictive_std,
    predictive_mean + 2 * predictive_std,
    alpha=0.2,
    label="Predictive ±2σ",
    color="green",
)
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[1].set_title("GP Regression via Conjugacy")

plt.tight_layout()
plt.show()


# #%%
# # computational time of likelihod training

# import time 

# results_summary = jnp.array([])
# delta_list = [0.1,0.2,0.25,0.5]

# rep=5
# for delta in delta_list:
#     time_total = 0
#     for i in range(rep):
#         # delta = 0.1
#         obs_noise = 0.1
#         obs_time = jnp.linspace(delta, T, int(T/delta ))
#         obs_idx = jnp.array([jnp.where(jnp.isclose(time_grid, t))[0][0] for t in obs_time])
#         obs_val = y[obs_idx] + jr.normal(key, obs_time.shape) * obs_noise
#         full_obs = jnp.stack([obs_time, obs_val])

#         # Define your negative log likelihood function.
#         # It should accept a parameter array and use the observations (full_obs).
#         def objective(params):
#             # Ensure params is a JAX array
#             params = jnp.array(params)
#             return compute_neg_log_marginal_likelihood(params, full_obs)

#         # Use JAX to compute the gradient of the objective.
#         grad_objective = jax.grad(objective)

#         # Choose an initial guess for the parameters.
#         initial_guess = [1, 1, 2]

#         start_time = time.time()
#         # Run the optimization (L-BFGS-B is a good choice for many problems).
#         result = minimize(objective, initial_guess, method='L-BFGS-B', jac=grad_objective)
#         end_time = time.time()

#         time_total = time_total + end_time - start_time

#     results_summary = jnp.concatenate([results_summary, jnp.array([obs_time.shape[0],time_total/rep])])

# results_summary = results_summary.reshape(len(delta_list),2)
# plt.plot(results_summary[:,0], results_summary[:,1],'x')
# plt.xlabel("Observation Number")
# plt.ylabel("Time (s)")
# plt.title("Computational Time of Likelihood Training for Kalman Filter GP Regression")
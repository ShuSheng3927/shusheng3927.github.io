#%%
import jax 
import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
import matplotlib.pyplot as plt
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
from gpjax.kernels.computations import DenseKernelComputation

jax.config.update("jax_enable_x64", True)

#%%

class brownian(gpx.kernels.AbstractKernel):
    variance: gpx.parameters.PositiveReal

    def __init__(
        self,
        variance: float = 1,
        active_dims: list[int] | slice | None = None,
        n_dims: int | None = None,
    ):
        super().__init__(active_dims, n_dims, DenseKernelComputation())
        self.variance = gpx.parameters.PositiveReal(jnp.array(variance), tag="variance")
    
    def __call__(
        self,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, ""]:
        K = (
            self.variance.value*jnp.minimum(x,y)
        )
        return K.squeeze()
     
#%%

kernel = brownian()
prior = gpx.gps.Prior(kernel=kernel, mean_function=gpx.mean_functions.Zero())

xx = jnp.linspace(0,3,200).reshape(-1,1)

key = jr.key(1)
sample_num = 30
samples = prior.predict(xx).sample(key,(sample_num,))
plt.figure(figsize=(8,4))
for sample in samples:
    plt.plot(xx.flatten(), sample,alpha=0.5)
plt.xlabel("t")
plt.ylabel("W_t")
plt.title("Sampled Brownian Motion Paths")
plt.show()

#%%
tt = jnp.arange(0,3.0001,0.2).reshape(-1,1)
traj = prior.predict(tt).sample(key,(1,))

plt.plot(tt.flatten(),traj[0], "x-", alpha=0.3, ms=5)

#%%
dataset = gpx.Dataset(X=tt, y=traj.reshape(-1,1))
likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n,obs_stddev=gpx.parameters.Static(0))
posterior = prior * likelihood

tt2 = jnp.arange(0,3.0001,0.1).reshape(-1,1)

traj2 = posterior.predict(tt2, train_data=dataset).sample(key, (1,))

plt.plot(tt.flatten(),traj[0], "x-",alpha=0.3, ms=5)
plt.plot(tt2, traj2.reshape(-1,1), "x-",alpha=0.3, ms=5)

#%%
dataset = gpx.Dataset(X = tt2, y=traj2.reshape(-1,1))
likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n,obs_stddev=gpx.parameters.Static(0))
posterior = prior * likelihood

tt3 = jnp.arange(0,3.0001,0.05).reshape(-1,1)

traj3 = posterior.predict(tt3, train_data=dataset).sample(key, (1,))

plt.figure(figsize=(8,4))
plt.plot(tt.flatten(),traj[0], "x-",alpha=0.5, ms=5, label="Stepsize = 0.2")
plt.plot(tt2, traj2.reshape(-1,1), "x-",alpha=0.5, ms=5, label="Stepsize = 0.1")
plt.plot(tt3, traj3.reshape(-1,1), "x-",alpha=0.5, ms=5, label="Stepsize = 0.05")
plt.legend()
plt.xlabel("t")
plt.ylabel("W_t")
plt.title("Upscale Brownian Motion Paths")
plt.show()
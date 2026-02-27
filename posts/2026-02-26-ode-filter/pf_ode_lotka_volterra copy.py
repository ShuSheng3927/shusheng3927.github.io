#%%
import numpy as np
import matplotlib.pyplot as plt
from probnum import filtsmooth, randvars, diffeq, randprocs, problems
from probnum.problems import TimeSeriesRegressionProblem
from scipy import stats

rng = np.random.default_rng(seed=123)

# Plotting
import matplotlib.pyplot as plt

# Consistent plotting styles for particles and for "true" latent states
particle_style = {
    "color": "C1",
    "marker": "o",
    "markersize": 5,
    "linestyle": "None",
    "alpha": 0.5,
}
latent_state_style = {"color": "C0", "linewidth": 5, "linestyle": "-", "alpha": 0.5}
# %%
def bern_rhs(t, x):
    return 1.5 * (x - x ** 3)


def bern_jac(t, x):
    return np.array([1.5 * (1 - 3 * x ** 2)])


t0, tmax = 0.0, 6.0
y0 = np.array([0.01])
bernoulli = problems.InitialValueProblem(t0=t0, tmax=tmax, y0=y0, f=bern_rhs, df=bern_jac)

def euler(f, t0, tmax, y0, dt):
    tt = np.arange(t0, tmax + dt, dt)
    ys = [y0]
    y_curr = y0
    for t in tt[1:]:
        y_next = y_curr + f(t, y_curr) * dt
        ys.append(y_next)
        y_curr = y_next
    return tt, np.array(ys)

# %%
dynamod = randprocs.markov.integrator.IntegratedWienerTransition(num_derivatives=2, wiener_process_dimension=1, forward_implementation="sqrt")

initmean = np.array([0.0, 0, 0.0])
initcov = 0.0125 * np.diag([1, 1.0, 1.0])
initrv = randvars.Normal(initmean, initcov, cov_cholesky=np.sqrt(initcov))

num_particles = 50
ode_prior = randprocs.markov.MarkovProcess(transition=dynamod, initrv=initrv, initarg=0.0)
importance = filtsmooth.particle.LinearizationImportanceDistribution.from_ekf(
    dynamod, backward_implementation="sqrt", forward_implementation="sqrt"
)

ode_pf = filtsmooth.particle.ParticleFilter(
    ode_prior,
    rng=rng,
    importance_distribution=importance,
    num_particles=num_particles,
)

num_locs = 50
data = np.zeros((num_locs, 1))
locs = np.linspace(0.0, tmax, num_locs)

info_op = diffeq.odefilter.information_operators.ODEResidual(num_prior_derivatives=2, ode_dimension=1)
ek1 = diffeq.odefilter.approx_strategies.EK1()

regression_problem = diffeq.odefilter.utils.ivp_to_regression_problem(
    ivp=bernoulli,
    locations=locs,
    ode_information_operator=info_op,
    approx_strategy=ek1,
    exclude_initial_condition=True,
    ode_measurement_variance=0.00001,
)

ode_posterior, _ = ode_pf.filter(regression_problem)
# %%

tt, ys = euler(bern_rhs, t0, tmax, y0, dt=0.01)

plt.plot(tt, ys[:, 0], "-", color='k', linewidth=2, alpha=0.8, label="True Solution")

plt.xlabel(r"$t$")
plt.ylabel(r"$y(t)$")
plt.plot(
    locs,
    ode_posterior.states.resample(rng=rng).support[:, :, 0],
    **particle_style,
    label="Particles"
)

# Remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()

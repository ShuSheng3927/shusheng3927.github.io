#%%
import matplotlib.pyplot as plt
import numpy as np
from probnum.diffeq import probsolve_ivp

rng = np.random.default_rng(seed=123)

#%%
def f(t, y):
    y1, y2 = y
    return np.array([0.5 * y1 - 0.05 * y1 * y2, -0.5 * y2 + 0.05 * y1 * y2])


def df(t, y):
    y1, y2 = y
    return np.array([[0.5 - 0.05 * y2, -0.05 * y1], [0.05 * y2, -0.5 + 0.05 * y1]])


t0 = 0.0
tmax = 20.0
y0 = np.array([20, 20])

#%%

def euler(f, t0, tmax, y0, dt):
    tt = np.arange(t0, tmax + dt, dt)
    ys = [y0]
    y_curr = y0
    for t in tt[1:]:
        y_next = y_curr + f(t, y_curr) * dt
        ys.append(y_next)
        y_curr = y_next
    return tt, np.array(ys)

sol = probsolve_ivp(
    f, t0, tmax, y0, df=df, step=0.1, adaptive=False, diffusion_model="dynamic"
)
means, stds = sol.states.mean, sol.states.std

tt, ys = euler(f, t0, tmax, y0, dt=0.1)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
ax1.plot(sol.locations, means[:, 0], label="x1", color='blue')
ax1.plot(sol.locations, means[:, 1], label="x2", color='orange')
ax1.plot(tt, ys[:, 0], label="x1 (Euler)", linestyle="--", color='blue')
ax1.plot(tt, ys[:, 1], label="x2 (Euler)", linestyle="--", color='orange')
ax1.legend()
ax2.fill_between(sol.locations, stds[:, 0], alpha=0.25, label="x1-unc.", color='blue')
ax2.fill_between(sol.locations, stds[:, 1], alpha=0.25, label="x2-unc.", color='orange')
ax2.legend()
fig.suptitle("EK0 Solution")
plt.show()

#%%
stepsizes = [0.1, 0.2, 0.3]   # <-- set list here
num_samples = 10

def run_and_plot(ax, method, stepsize, show_legend_labels=False):
    sol = probsolve_ivp(
        f,
        t0,
        tmax,
        y0,
        df=df,
        method=method,
        step=stepsize,
        algo_order=1,
        adaptive=False,
        diffusion_model="dynamic",
    )
    tt, ys = euler(f, t0, tmax, y0, dt=0.0001)

    locations = sol.locations
    samples = sol.sample(rng=rng, t=locations, size=num_samples)

    solution = sol(locations)
    means = solution.mean
    stds = solution.std

    # ---- COLORS (shared for samples + CI) ----
    c1 = "C0"   # x1
    c2 = "C1"   # x2

    # Samples
    for i, sample in enumerate(samples):
        lab_x1 = "x1 Traj Sample" if (show_legend_labels and i == 0) else None
        lab_x2 = "x2 Traj Sample" if (show_legend_labels and i == 0) else None

        ax.plot(
            locations,
            sample[:, 0],
            "--",
            color=c1,
            linewidth=1,
            alpha=0.5,
            label=lab_x1,
        )
        ax.plot(
            locations,
            sample[:, 1],
            "--",
            color=c2,
            linewidth=1,
            alpha=0.5,
            label=lab_x2,
        )

    # Means (black, as before)
    ax.plot(
        locations,
        means[:, 0],
        '--',
        color="k",
        label="x1 Traj Mean" if show_legend_labels else None,
    )
    ax.plot(
        locations,
        means[:, 1],
        '--',
        color="k",
        label="x2 Traj Mean" if show_legend_labels else None,
    )
    ax.plot(tt, ys[:, 0], "-", color='k', linewidth=2, alpha=0.8, label="x1 Euler", zorder=10 if show_legend_labels else None)
    ax.plot(tt, ys[:, 1], "-", color='k', linewidth=2, alpha=0.8, label="x2 Euler", zorder=10 if show_legend_labels else None)

    # 2-sigma bands — SAME COLORS
    ax.fill_between(
        locations,
        means[:, 0] - 2 * stds[:, 0],
        means[:, 0] + 2 * stds[:, 0],
        color=c1,
        alpha=0.1,
        label=r"x1 Traj 2$\sigma$" if show_legend_labels else None,
    )

    ax.fill_between(
        locations,
        means[:, 1] - 2 * stds[:, 1],
        means[:, 1] + 2 * stds[:, 1],
        color=c2,
        alpha=0.1,
        label=r"x2 Traj 2$\sigma$" if show_legend_labels else None,
    )

    # Observation points
    ax.plot(
        sol.locations,
        sol.states.mean,
        "o",
        ms=4,
    )

    return sol

n = len(stepsizes)
fig, axes = plt.subplots(
    nrows=n,
    ncols=2,
    sharex=True,
    sharey=True,
    figsize=(16, 4 * n),
)

# If n==1, axes won't be 2D; normalize it
if n == 1:
    axes = np.array([axes])

for r, h in enumerate(stepsizes):
    ax0, ax1 = axes[r, 0], axes[r, 1]

    # Only collect legend entries from the first row (avoid duplicates)
    show_labels = (r == 0)

    run_and_plot(ax0, method="EK0", stepsize=h, show_legend_labels=show_labels)
    run_and_plot(ax1, method="EK1", stepsize=h, show_legend_labels=show_labels)

    # Column titles only on top row
    if r == 0:
        ax0.set_title("Extended Kalman Filter Order 0", fontsize=16)
        ax1.set_title("Extended Kalman Filter Order 1", fontsize=16)

    # Stepsize label on the left side of the row (as ylabel)
    ax0.set_ylabel(
    f"Stepsize={h}",
    fontsize=16,
    rotation=90,
    labelpad=25,
    va="center",
    fontweight="bold",
    bbox=dict(
        boxstyle="square",
        edgecolor="black",
        facecolor="white",
        linewidth=1.5,
    ),
    )

    ax0.tick_params(axis="both", labelsize=12)
    ax1.tick_params(axis="both", labelsize=12)

# One shared legend (from first-row, second axis — or any axis)
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=15)

fig.subplots_adjust(bottom=0.12, hspace=0.15, wspace=0.05)
fig.suptitle("Probabilistic Solution to Lotka-Volterra", fontsize=20, y = 0.95, fontweight="bold")

plt.show()

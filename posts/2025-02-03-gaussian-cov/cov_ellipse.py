#%%
import numpy as np
import matplotlib.pyplot as plt

def covariance_mat(var1, var2, cor):
    var12 = var1 * var2 * cor
    return np.array([[var1, var12], [var12, var2]])

def cov_ellipse(cov):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    axis_lengths = np.sqrt(eigenvalues)
    theta = np.linspace(0, 2 * np.pi, 500)
    circle = np.array([np.cos(theta), np.sin(theta)])  # Shape (2, N)
    ellipse = eigenvectors @ (axis_lengths[:, None] * circle)  # Shape (2, N)
    return ellipse

def entropy(cov):
    D = cov.shape[0]
    return 0.5*(D*np.log(np.pi) + D + np.log(np.linalg.det(cov)))

#%%
cov1 = covariance_mat(1.,1.,0.)
cov2 = covariance_mat(1.,1.,0.5)
cov3 = covariance_mat(1.,2.,0.)
cov4 = covariance_mat(1.,2.,0.5)
covs = [[cov1, cov2],[cov3,cov4]]

#%%
fig, ax = plt.subplots(2,2,figsize=(12,12))
for i in range(2):
    for j in range(2):
        ax[i,j].plot(cov_ellipse(covs[i][j])[0], cov_ellipse(covs[i][j])[1])
        ax[i,j].set_xlim(-2,2)
        ax[i,j].set_ylim(-2,2)
        ax[i,j].set_title(f'Trace={np.trace(covs[i][j]).astype(float)}; Determinant={np.linalg.det(covs[i][j]).astype(float)}; \n Entropy={int(entropy(covs[i][j])*100)/100}; Frobenius Norm={int(np.linalg.norm(covs[i][j])*100)/100}')
        ax[i,j].text(0, 0, f'{covs[i][j]}', ha='center', va='center', fontsize=12)        
fig.show()



Implementation Guide for cskpd_new Library
This document provides a comprehensive guide on how to implement and use the cskpd_new library. Below is the same information presented in markdown format:

1. Installation and Loading
First, ensure you have installed the required packages:

pip install statsmodels numpy pandas matplotlib seaborn
Load the necessary modules from our library:

from cskpd_new import *
2. Core Components of CSKPD Class
2.1 Initialization Parameters
The 
CSKPD
 class is initialized with several key parameters:

p_list: A list containing tuples specifying the dimensions (M, N) for each Kronecker factor.
lam: A vector of regularization parameters.
g: The link function used in the model.
n_cores: Number of CPU cores to utilize during computations.
max_iter: Maximum number of iterations allowed for convergence.
print_iter: Interval at which progress prints are displayed.
2.2 Key Methods
The class provides two main methods:

fit()
: Fits the model using your data.
custom_cross_validate()
: Performs custom cross-validation with specified scorers (e.g., MSE, AUC).
3. Example Implementation
Here's an example of how to create and use a 
CSKPD
 model:

# Define parameters
p_list = [[16, 16], [32, 32]]
lam = [2**i for i in range(5, 6)]  # Regularization parameters
R = 3  # Number of components
g = Logit()  # Link function

# Create model instance
test_model = CSKPD(p_list=p_list, lam=lam, R=R, g=g, n_cores=12,
                    max_iter=3000, print_iter=50)

# Fit the model to your data
test_model.fit(X, Y, tol=1e-3)  # X: Input tensor, Y: Response vector

# View model attributes
print(test_model.grid_values.keys())
4. Cross-Validation Usage
The 
custom_cross_validate
 method allows you to evaluate the model's performance using custom scorers:

# Define scorers
scorers = {
    'MSE': mean_squared_error,
    'AUC': auc
}

# Perform cross-validation
t1 = test_model.custom_cross_validate(X, Y, scorers=scorers)

# Access validation results
for ti in t1:
    print(ti['results']['test_AUC'])

# Visualize results using custom plotting functions
plt.figure(figsize=(12, 4))
ax = plt.subplot(1, 3, 1)
sns.heatmap(fun_normalization(test_model.C), cmap="rainbow", cbar=False)
ax.set_xticks([])
ax.set_yticks([])
ax = plt.subplot(1, 3, 2)
sns.heatmap(fun_normalization(t1[np.argmax(t1[4]['results'])]['model'].C), 
            cmap="rainbow", cbar=True)
ax.set_xticks([])
ax.set_yticks([])
plt.show()
5. Parameter Explanation
5.1 Averaging Across Components
The following code demonstrates how to compute the final estimate (C) by averaging across components:

R = test_model.grid_values['R']
# Initialize C list
C = []

for i in range(R):
    # Extract component-specific parameters
    A = test_model.grid_values['A'][:, i].reshape(test_model.grid_values['p'])
    B = test_model.grid_values['B'][:, i].reshape((len(lam), len(lam)))
    
    # Compute Kronecker product for this component
    C_kron = np.kron(A, B)
    
    # Store component parameters and output
    test_model.grid_values['C'].append(C_kron)
    print(test_model.grid_values['B'])
5.2 Visualization of Components
The following code generates heatmaps to visualize the components:

plt.figure(figsize=(12, 4))
ax = plt.subplot(1, 3, 1)
sns.heatmap(fun_normalization(C[0]), cmap="rainbow", cbar=False)
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(1, 3, 2)
sns.heatmap(fun_normalization(C[1]), cmap="rainbow", cbar=False)
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(1, 3, 3)
sns.heatmap(fun_normalization(C[2]), cmap="rainbow", cbar=False)
ax.set_xticks([])
ax.set_yticks([])
plt.show()
6. Sample Output
Example Output from 
custom_cross_validate
:
{
    'params': {
        'R': 3,
        'lam': [1024, 2048]
    },
    'train_scores': {
        'MSE': [0.052, 0.067, ...],
        'AUC': [0.89, 0.91, ...]
    },
    'test_scores': {
        'MSE': [0.043, 0.058, ...],
        'AUC': [0.87, 0.89, ...]
    }
}
7. Final Notes
This implementation guide provides a solid foundation for using the cskpd_new library. For more detailed information on specific parameters and methods, refer to our official documentation or repository.

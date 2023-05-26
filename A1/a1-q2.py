import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

np.random.seed(42)  # set for reprod

# Init variables
num_samples = 1000
mean = np.array([1, 0.5])
mean[1] = np.radians(mean[1])

cov = np.array([[0.01, 0], [0, 1.00]])
cov[1, 1] = np.radians(cov[1, 1])

# Constructed this way to reduce errors and improve readability
def cov_y_builder(mean, cov):
   rho = mean[0]
   theta = mean[1]

   a = np.cos(theta)
   b = -rho * np.sin(theta)
   c = np.sin(theta)
   d = rho * np.cos(theta)

   e = cov[0, 0]
   f = cov[0, 1]
   g = cov[1, 0]
   h = cov[1, 1]

   cov_y = np.zeros((2, 2))
   cov_y[0, 0] = (a*e+b*g)*a+(a*f+b*h)*b
   cov_y[0, 1] = (a*e+b*g)*c+(a*f+b*h)*d
   cov_y[1, 0] = (c*e+d*g)*a+(c*f+d*h)*b
   cov_y[1, 1] = (c*e+d*g)*c+(c*f+d*h)*d
   return cov_y

# Generate samples
polar_samples = np.random.multivariate_normal(mean, cov, num_samples)

# Create the polar to cartesian transform and perform it
transform_v = np.vectorize(lambda rho, theta: (rho * np.cos(theta), rho * np.sin(theta)))

cartesian_mean = transform_v(mean[0], mean[1])
cartesian_coords = transform_v(polar_samples[:, 0], polar_samples[:, 1])

# Compute the eigenvalues and eignevectors of the transformed covariance
cov_y = cov_y_builder(mean, cov)
eigenvalues, eigenvectors = np.linalg.eig(cov_y)

# Create the uncertaintly ellipse

# Simple confidence variant
s = np.sqrt(-2 * np.log(1 - 0.95)) # simple 
major = 2 * np.sqrt(np.max(eigenvalues)) * s
minor = 2 * np.sqrt(np.min(eigenvalues)) * s

# chi-squared variant (used in submission)
# s = 5.991 # 95% confidence
# major = 2 * np.sqrt(np.max(eigenvalues) * s) 
# minor = 2 * np.sqrt(np.min(eigenvalues) * s)

max_eigenvalue_index = np.argmax(eigenvalues)
angle = -np.degrees(np.arctan2(eigenvectors[max_eigenvalue_index,1], eigenvectors[max_eigenvalue_index,0]))
ellipse = Ellipse(xy=cartesian_mean, width=major, height=minor, angle=angle, alpha=0.3, color='r', fill=False)

# Plotting
sns.scatterplot(x=cartesian_coords[0], y=cartesian_coords[1], s=3)
plt.gca().add_patch(ellipse)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('')

plt.axis('equal')
# Center on origin
# Comment out to center on the mean
plot_scale = 1.5
plt.xlim(-plot_scale, plot_scale)
plt.ylim(-plot_scale, plot_scale)

plt.show()
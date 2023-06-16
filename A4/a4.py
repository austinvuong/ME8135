import numpy as np
import random
import matplotlib.pyplot as plt

# Based on: https://www.cs.toronto.edu/~lindell/teaching/420/slides/lecture8.pdf

# np.random.seed(42)

# hardcoding for ease
points_source = np.array([
   [1.90659, 2.51737],
   [2.20896, 1.1542 ],
   [2.37878, 2.15422],
   [1.98784, 1.44557],
   [2.83467, 3.41243],
   [9.12775, 8.60163],
   [4.31247, 5.57856],
   [6.50957, 5.65667],
   [3.20486, 2.67803],
   [6.60663, 3.80709],
   [8.40191, 3.41115],
   [2.41345, 5.71343],
   [1.04413, 5.29942],
   [3.68784, 3.54342],
   [1.41243, 2.6001 ],
])
points_destination = np.array([
   [5.0513 , 1.14083],
   [1.61414, 0.92223],
   [1.95854, 1.05193],
   [1.62637, 0.93347],
   [2.4199 , 1.22036],
   [5.58934, 3.60356],
   [3.18642, 1.48918],
   [3.42369, 1.54875],
   [3.65167, 3.73654],
   [3.09629, 1.41874],
   [5.55153, 1.73183],
   [2.94418, 1.43583],
   [6.8175 , 0.01906],
   [2.62637, 1.28191],
   [1.78841, 1.0149 ],
])

def homography(points_source, points_destination):
   A = []

   for i in range(0, len(points_source)):
      x0, y0 = points_source[i][0], points_source[i][1]
      x1, y1 = points_destination[i][0], points_destination[i][1]
      A.append([x0, y0, 1, 0, 0, 0, -x1*x0, -x1*y0, -x1])
      A.append([ 0,  0, 0,x0,y0, 1, -y1*x0, -y1*y0, -y1])

   _, _, V = np.linalg.svd(A)
   return np.reshape(V[-1], (3, 3))

def apply_homography(H, points):
   homogenized_points = np.column_stack((points, np.ones(points.shape[0])))
   transformed_points = (H @ homogenized_points.T).T
   transformed_points = transformed_points[:, :2] / transformed_points[:, 2:] # normalize and drop the 1
   return transformed_points

# note: if min_points is too large this will not terminiate
# for the assigment this is fine
def ransac_homography(points_source, points_destination, threshold=0.005, min_points=10):
   H = np.zeros(3)
   inliers_indexes = []

   while len(inliers_indexes) < min_points:
      choice = np.random.choice(len(points_source), 4, replace=False) # using the minimum needed points
      sample_source = points_source[choice]
      sample_destination = points_destination[choice]

      H = homography(sample_source, sample_destination)
      transformed_points = apply_homography(H, points_source)

      errors = np.linalg.norm(points_destination - transformed_points, axis=1)
      inliers_indexes = np.where(errors < threshold)[0]

   return H, inliers_indexes

# Find the inliers and outliers
H, inliers_indexes = ransac_homography(points_source, points_destination)
outliers_index = np.setdiff1d(np.arange(len(points_source)), inliers_indexes)

# Normalize and print the transform for the report
# H = H/H[2, 2]
# print(H)
# Pasted for easy access (it's consistent)
# [[0.70917868 0.43442468 0.129146  ] 
#  [0.28310411 0.17443173 0.42877658] 
#  [0.15491603 0.01687054 1.        ]]

# Plot inliers
plt.scatter(points_source[inliers_indexes, 0], points_source[inliers_indexes, 1], color='blue', marker='o')
plt.scatter(points_destination[inliers_indexes, 0], points_destination[inliers_indexes, 1], color='red', marker='o')

# Plot outliers
plt.scatter(points_source[outliers_index, 0], points_source[outliers_index, 1], color='lightblue', marker='x')
plt.scatter(points_destination[outliers_index, 0], points_destination[outliers_index, 1], color='pink', marker='x')

# Plot the line connecting inliers
for i in inliers_indexes:
   plt.plot([points_source[i, 0], points_destination[i, 0]], 
            [points_source[i, 1], points_destination[i, 1]], 
            color='lightgray')

plt.title('Homography')
plt.xlabel('x')
plt.ylabel('y')

plt.show()


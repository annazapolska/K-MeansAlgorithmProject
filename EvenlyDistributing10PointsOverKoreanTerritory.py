"""# **Task 2**"""

from shapely.geometry import Polygon, Point
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming territory_data is a list of (x, y) coordinates forming a polygon
territory_data = pd.read_csv('/content/drive/MyDrive/IntML/Project data/South_Korea_territory.csv')

import seaborn as sns

sns.set(rc={'figure.figsize':(8,12)})
sns.scatterplot(x='Longitude (deg)', y='Latitude (deg)', data=territory_data)

territory_data = territory_data.to_numpy()

# Create a Polygon object representing the country
country_polygon = Polygon(territory_data)

# Determine the bounding box of the country
min_x, min_y = np.min(territory_data[:, 0]), np.min(territory_data[:, 1])
max_x, max_y = np.max(territory_data[:, 0]), np.max(territory_data[:, 1])

# Generate 5000 random points within the country
points_within_country = []
while len(points_within_country) < 5000:
    random_x = random.uniform(min_x, max_x)
    random_y = random.uniform(min_y, max_y)
    random_point = Point(random_x, random_y)
    if random_point.within(country_polygon):
        points_within_country.append((random_x, random_y))

# Convert points_within_country to a NumPy array
points_within_country = np.array(points_within_country)

# Plot the territory points
territory_x, territory_y = zip(*territory_data)
plt.plot(territory_x, territory_y, 'b-', label='Territory Points')

# Plot the randomly generated points
random_x, random_y = zip(*points_within_country)
plt.plot(random_x, random_y, 'ro', label='Random Points')
plt.title('Territory Points and Random Points')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.legend()
plt.show()


def k_means(data, k, max_iterations=20000):
    #set the count to 0
    iteration_count = 0

    #randomly generate initial centroids
    centroids = data[np.random.choice(range(len(data)), size=k, replace=False)]

    for iteration in range(max_iterations):
        #for each of the points, calculate the distance to each of the centroids
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        #assign the points to centroids to which they are the closest
        labels = np.argmin(distances, axis=1)

        #generate new centroids by taking the mean of all points in the cluster label
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        #stopping criteria. Algorithm stops if centroids did not move compared to previous iteration
        if np.all(centroids == new_centroids):
            break
        
        #asign to centroids newly generated values
        centroids = new_centroids
        iteration_count = iteration + 1 #increase the count

    return labels, centroids, iteration_count


#initialize K parameter and run the algorithm
k = 10
labels, centroids, iteration_count = k_means(points_within_country, k)

#Print number of iterations and and the resulting clusters and centroids
print("Number of iterations:", iteration_count) 
colors = ['r', 'g', 'b']
plt.figure(figsize=(8, 12))
for i in range(k):
    cluster_data = points_within_country[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label='Cluster {}'.format(i+1))
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=200, label='Centroids')
#print territory boundaries
territory_x, territory_y = zip(*territory_data)
plt.plot(territory_x, territory_y, 'b-', label='Territory Points')
plt.legend(bbox_to_anchor=[1.22,1.0])
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.show()

"""# **Task 3**"""

vertiport_candidates = pd.read_csv('/content/drive/MyDrive/IntML/Project data/Vertiport_candidates.csv')

#visualizing the Vertiport Candidates
sns.scatterplot(x='Longitude (deg)', y = 'Latitude (deg)', data=vertiport_candidates )

#Task 3 Approach 1
#generating evenly distributed 17 centroids to use the as initial centroids 

from shapely.geometry import Polygon, Point
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming territory_data is a list of (x, y) coordinates forming a polygon
territory_data = pd.read_csv("/content/drive/MyDrive/IntML/Project data/South_Korea_territory.csv").to_numpy()

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


def k_means(data, k, max_iterations=5000):
    iteration_count = 0
    
    #randomly initialize centroids
    centroids = data[np.random.choice(range(len(data)), size=k, replace=False)]

    #repeat the algorithm until max_iterations is reached 
    for iteration in range(max_iterations):
        #calculate the distance from each point to each centroid and assign point to closest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        #generate new centroids of the formed clusters by taking a mean of all poits in a cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        #terminate algorithm is the location of centroids did not change
        if np.all(centroids == new_centroids):
            break

        #assign new centroid values to centroids and increase count
        centroids = new_centroids
        iteration_count = iteration + 1

    return labels, centroids, iteration_count

#initlialize K parameter and start the algorithm
k = 17
labels, centroids, iteration_count = k_means(points_within_country, k)


#set the colors for the clusters
colors2 = ['darkgrey','brown','red','salmon','peru','burlywood','darkorange','darkgoldenrod','olive','lawngreen','forestgreen','aquamarine','lightseagreen',
           'teal','cyan','deepskyblue','blue','royalblue','blueviolet','indigo','fuchsia','deeppink','crimson','chocolate','yellow','lime']

#print the number of iterations and resulting clusters
print("Number of iterations:", iteration_count)
plt.figure(figsize=(8, 12))
for i in range(k):
    cluster_data = points_within_country[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label='Cluster {}'.format(i+1),c=colors2[i])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=200, label='Centroids')
plt.legend(bbox_to_anchor=[1.05,1.0])
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.show()
print(centroids)

#Task 3
#Generating centroids on Vertiport Candidates using evenly distributed centroids from last block as initial centroids
epsilon = 0.0000001  # convergence tolerance

#convert vertiport candidates array into numpy array
vertiport_candidates = vertiport_candidates.to_numpy()

#set initial centroids to the evenly distributed centroids generated in the last block
initial_centroids = np.copy(centroids)
print(initial_centroids)

def k_means(data, k, intial_cantroids, max_iterations=15000):
    #check if the number of centroids coresponds to the needed number of initial centorids
    if len(initial_centroids) != k:
        raise ValueError("Number of initial centroids must be equal to k")

    #set centroids to initial centroids and set count to 0
    centroids = initial_centroids
    iteration_count = 0

    #run the algorithm for the maximum number of iterations unless the stoppping criteria is reached
    for iteration in range(max_iterations):
        #calculate the distance from each point to each centroid and assign point to closest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        #generate new centroids of the formed clusters by taking a mean of all poits in a cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        #terminate algorithm is the location of centroids changed less than the epsilon value
        if np.all(abs(new_centroids - centroids) < epsilon):
            break
        
        #assign new centroid values to centroids and increase count
        centroids = new_centroids
        iteration_count = iteration + 1

    return labels, centroids, iteration_count

#initlialize K parameter and start the algorithm
k = 17
labels, centroids, iteration_count = k_means(vertiport_candidates, k, initial_centroids)



#set the colors for the clusters
colors2 = ['darkgrey','brown','red','salmon','peru','burlywood','darkorange','darkgoldenrod','olive','lawngreen','forestgreen','aquamarine','lightseagreen',
           'teal','cyan','deepskyblue','blue','royalblue','blueviolet','indigo','fuchsia','deeppink','crimson','chocolate','yellow','lime']

#print the number of iterations and resulting clusters
print("Centroids:", centroids)
print("Number of iterations:", iteration_count)
plt.figure(figsize=(8,12))
for i in range(k):
      cluster_data = vertiport_candidates[labels == i]
      plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label='Cluster {}'.format(i+1),c=colors2[i])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=200, label='Centroids')
plt.plot(territory_data[:,0], territory_data[:,1])
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('K-Means Clustering')
plt.legend(bbox_to_anchor=[1.05,1.0])
plt.show()

#Task3 No Centroids given (Initial Centroids generated randomly)
epsilon = 0.0000001  # convergence tolerance


def k_means(data, k, max_iterations=100):
    #set centroids to initial centroids and set count to 0
    centroids = data[np.random.choice(range(len(data)), size=k, replace=False)]
    iteration_count = 0

    #run the algorithm for the maximum number of iterations unless the stoppping criteria is reached
    for iteration in range(max_iterations):
        #calculate the distance from each point to each centroid and assign point to closest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        #generate new centroids of the formed clusters by taking a mean of all poits in a cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        #terminate algorithm is the location of centroids changed less than the epsilon value
        if np.all(abs(new_centroids - centroids) < epsilon):
            break

        #assign new centroid values to centroids and increase count
        centroids = new_centroids
        iteration_count = iteration + 1

    return labels, centroids, iteration_count



#initlialize K parameter and start the algorithm
k = 17
labels, centroids, iteration_count = k_means(vertiport_candidates, k)


# Setting the colors for centroids
colors2 = ['darkgrey','brown','red','salmon','peru','burlywood','darkorange','darkgoldenrod','olive','lawngreen','forestgreen','aquamarine','lightseagreen',
           'teal','cyan','deepskyblue','blue','royalblue','blueviolet','indigo','fuchsia','deeppink','crimson','chocolate','yellow','lime']

#print the number of iterations and resulting clusters
print("Centroids:", centroids)
print("Number of iterations:", iteration_count)
plt.figure(figsize=(8,12))
for i in range(k):
      cluster_data = vertiport_candidates[labels == i]
      plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label='Cluster {}'.format(i+1), c=colors2[i])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=200, label='Centroids')
plt.plot(territory_data[:,0], territory_data[:,1])
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('K-Means Clustering')
plt.legend(bbox_to_anchor=[1.05,1.0])
plt.show()

from sklearn.cluster import KMeans

distortions = []
K = range(1,50)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(vertiport_candidates)
    distortions.append(kmeanModel.inertia_)

#Using Elbow method for optimal number of Vertiports

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#Task3 No Initil points
#Optimal Number of Vertiports - 26


#setting the colors for clusters
colors2 = ['darkgrey','brown','red','salmon','peru','burlywood','darkorange','darkgoldenrod','olive','lawngreen','forestgreen','aquamarine','lightseagreen',
           'teal','cyan','deepskyblue','blue','royalblue','blueviolet','indigo','fuchsia','deeppink','crimson','chocolate','yellow','lime']
# setting a convergence tolerance
epsilon = 0.0000001  


def k_means(data, k, max_iterations=100):
    #set centroids to initial centroids and set count to 0
    centroids = data[np.random.choice(range(len(data)), size=k, replace=False)]
    iteration_count = 0

    #run the algorithm for the maximum number of iterations unless the stoppping criteria is reached
    for iteration in range(max_iterations):
        #calculate the distance from each point to each centroid and assign point to closest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        #generate new centroids of the formed clusters by taking a mean of all poits in a cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        #terminate algorithm is the location of centroids changed less than the epsilon value
        if np.all(abs(new_centroids - centroids) < epsilon):
            break
        
        #assign new centroid values to centroids and increase count
        centroids = new_centroids
        iteration_count = iteration + 1

    return labels, centroids, iteration_count



#initlialize K parameter and start the algorithm
k = 26
labels, centroids, iteration_count = k_means(vertiport_candidates, k)


#print the number of iterations and resulting clusters
print("Centroids:", centroids)
print("Number of iterations:", iteration_count)
plt.figure(figsize=(8,12))
for i in range(k):
      cluster_data = vertiport_candidates[labels == i]
      plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label='Cluster {}'.format(i+1), c=colors2[i])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=200, label='Centroids')
plt.plot(territory_data[:,0], territory_data[:,1])
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('K-Means Clustering')
plt.legend(bbox_to_anchor=[1.05,1.0])
plt.show()

import numpy as np
import emlib as em
import sys
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d


HULL_CUTOFF = 0.4

if len(sys.argv)!=4:
  print "usage: python gmm.py k n stop"
  print "k = number of clusters"
  print "n = number of alg iterations"
  print "stop = the delta stopping condition"
  sys.exit()

#Parse command line args
k = int(sys.argv[1])
n = int(sys.argv[2])
s = float(sys.argv[3])

#Parse the data into a 2xN np array
with open("GMM_dataset_546.txt") as data_file:
  data = np.array([])
  for row in data_file:
    temp = np.asarray(row.split())
    data = np.append(data, temp.astype(np.float))
  data = data.reshape(int(len(data)/2), 2)

#Max log-likelihood
max_ll = 0

for i in range(n):
  print "Running k-means for 2 iterations..."
  #Run k means and get the centers and wcss error
  (centers, err) = em.kmeans(data, k, 1)

  #Classify each point for graphing
  classes = em.classify(data, centers, k)

  #Run our GMM algorithm
  print "Running GMM on k-means init..."
  (weights, centers, likelihood) = em.gmm(data, centers, classes, s)

  if likelihood > max_ll:
    max_ll = likelihood
    m_weights = weights
    m_centers = centers

centers = m_centers
weights = m_weights

#Create a colormap for each k value
gradient = np.linspace(0, 1, k)
cmap = plt.cm.get_cmap('rainbow')
colors = np.zeros((k,4))
labels = []
for i in range(k):
  colors[i] = cmap(gradient[i])
  labels.append("Cluster "+str(i+1))
pt_colors = np.transpose(em.get_colors(colors, weights))

#Plot the data
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=pt_colors, s=10, alpha=1)

#Plot the centroids and hulls
for i in range(k):
  ax.scatter(centers[:, 0], centers[:, 1], c=colors, s=70, label=labels[i], alpha=1, edgecolors = 'k')
  #Get the hulls
  points = []
  for j in range(len(data)):
    if weights[j][i] > HULL_CUTOFF:
      points.append(data[j])
  points = np.asarray(points)
  hull = ConvexHull(points)
 
  #Plot the hulls
  for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], c=colors[i])

ax.grid(True)
ax.set_title('k = {}; n = {}; s = {}'.format(k,n,s))
ax.text(0.9, 0.05, str(round(err,2)), horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
plt.savefig('k{}/n-{}_s-{}.png'.format(k, n, s))
plt.show()
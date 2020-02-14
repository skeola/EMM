import numpy as np
import fcmlib as fcm
import sys
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

HULL_CUTOFF = 0.4

if len(sys.argv)!=5:
  print("usage: python3 kmeans.py c r m e")
  print("c = number of clusters")
  print("r = number of alg iterations")
  print("m = 'fuzzifier' parameter")
  print("e = sensitivity of stopping condition")
  sys.exit()

#Parse command line args
c = int(sys.argv[1])
r = int(sys.argv[2])
m = float(sys.argv[3])
sens = float(sys.argv[4])

#Parse the data into a 2xN np array
with open("GMM_data_fall2019.txt") as data_file:
  data = np.array([])
  for row in data_file:
    temp = np.asarray(row.split())
    data = np.append(data, temp.astype(np.float))
  data = data.reshape(int(len(data)/2), 2)


#Run k means and get the centers and wcss error
(weights, err) = fcm.fcm(data, c, r, m, sens)

#Classify each point for graphing
centers = fcm.get_centroids(data, c, weights, m)


#Output data to stdout
print("centroids:\n", centers)
print("wcss error: ",err)


#Create a colormap for each k value
gradient = np.linspace(0, 1, c)
cmap = plt.cm.get_cmap('rainbow')
colors = np.zeros((c,4))
labels = []
for i in range(c):
  colors[i] = cmap(gradient[i])
  labels.append("Cluster "+str(i+1))
pt_colors = np.transpose(fcm.get_colors(colors, weights))

#Plot the data
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=pt_colors, s=10, alpha=1)

#Plot the centroids and hulls
for i in range(c):
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
ax.set_title('c = {}; r = {}; m = {}'.format(c,r,m))
ax.text(0.9, 0.05, str(round(err,2)), horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
plt.savefig('c{}/r-{}_m-{}.png'.format(c, r, m))
plt.show()

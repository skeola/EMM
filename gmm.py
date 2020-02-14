import numpy as np
import emlib as em
import sys
import matplotlib.pyplot as plt

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

print "Running k-means for 2 iterations..."
#Run k means and get the centers and wcss error
(centers, err) = em.kmeans(data, k, 1)

#Classify each point for graphing
classes = em.classify(data, centers, k)

#Output data to stdout
print "centers after one iteration: {}".format(centers)

#Run our GMM algorithm
(classes, centers) = em.gmm(data, centers, classes, s)

#Create a colormap for each k value
gradient = np.linspace(0, 1, k)
cmap = plt.cm.get_cmap('rainbow')
colors = np.zeros((k,4))
labels = []
for i in range(k):
  colors[i] = cmap(gradient[i])
  labels.append("Cluster "+str(i+1))

#Plot the data and the centers
fig, ax = plt.subplots()
for i in range(len(data)):
  ax.scatter(x=data[i][0], y=data[i][1], c=colors[int(classes[i])], s=10, alpha=0.5)

for i in range(k):
  ax.scatter(centers[i][0], centers[i][1], c='k', s=50, label=labels[i], alpha=1)

ax.grid(True)
ax.set_title('k = {}; r = {}'.format(k,n))
plt.savefig('k{}/r-{}_e-{}.png'.format(k,n,round(err[0],2)))
plt.show()
exit()
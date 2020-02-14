import numpy as np
from scipy.stats import multivariate_normal
###############
##--K-MEANS--##
###############
#Run the kmeans algorithm on the data for k clusters, n times
#Return the cluster centers and wcss error for the best run
def kmeans(data, k, n):
  #Epoch Loop
  for i in range(n):
    print "EPOCH {}:".format(i+1)
    #Get the initial random centers
    centers = init(data, k)
    wcss = []
    prev_centers = centers - 1
    #Repeat training until the centers stop moving
    while np.array_equal(prev_centers,centers) == False:
      prev_centers = np.copy(centers)
      (centers, wcss) = learn(data, k, centers)
    print "WCSS Error: {}".format(wcss)  
    #Training is done, compare to previous error minimum
    if i==0:
      (m_cen, m_err) = (centers, wcss)
    else:
      if wcss < m_err:
        (m_cen, m_err) = (centers, wcss)
    print "min error: {}".format(m_err)
  #Return results with minimum wcss error
  return (m_cen, m_err)
  
######################
##--INITIALIZATION--##
######################
#Do preprocessing on the data and centers
def init(data, k):
  #Find the min and max of x and y values in the inputs
  xmin = data[0][0]
  xmax = data[0][0]
  ymin = data[0][1]
  ymax = data[0][1]

  #Iterate through the data
  for dp in data:
    if dp[0] > xmax:
      xmax = dp[0]
    else:
      if dp[0] < xmin:
        xmin = dp[0]
    if dp[1] > ymax:
      ymax = dp[1]
    else:
      if dp[1] < ymin:
        ymin = dp[1]

  #Use the min/max to pick k random points between them
  centers = np.empty([k, 2])
  for i in range(0, k):
    centers[i][0] = np.random.random()*(xmax-xmin)+xmin
    centers[i][1] = np.random.random()*(ymax-ymin)+ymin
  
  return centers

################
##--LEARNING--##
################
#Iterate through the data until the centers stop moving
def learn(data, k, centers):
  #Initialize containers
  distance = np.empty([k, 1])
  err = 0
  
  #Instead of holding each point belonging to the cluster
  #we will hold the sum of the points and a count as a tuple
  cluster = np.array([])
  for i in range(k):
    cluster = np.append(cluster, (np.array([0,0]), 0), axis = 0)
  cluster = cluster.reshape((k,2))
  #Iterate through each point
  for dp in data:
    #For each center, calculate the distance
    for i in range(0, k):
      distance[i] = np.sqrt((dp[0] - centers[i][0])**2 + (dp[1] - centers[i][1])**2)
    #'Assign' the point to a cluster
    class_id = np.argmin(distance)
    cluster[class_id] += [dp, 1]
    #The distance from the point to its class is the wcss error
    #Save it now since we did the calculation already
    err += distance[class_id]
 
  #Adjust the cluster centers
  for i in range(k):
    if cluster[i][1] != 0:
      centers[i] = np.divide(cluster[i][0],cluster[i][1])

  return (centers, err)

  
################
##--CLASSIFY--##
################
def classify(data, centers, k):
  #Classification of each point from 0-k
  classification = np.zeros((len(data),1))
 
  #Init variables
  distance = np.empty([k, 1])
  count = 0

  #Iterate through each point
  for dp in data:
    #For each center, calculate the distance
    for i in range(0, k):
      distance[i] = np.sqrt((dp[0] - centers[i][0])**2 + (dp[1] - centers[i][1])**2)
    #Assign the point to a cluster
    classification[count] = np.argmin(distance)
    count += 1
  
  return classification







#############
##---GMM---##
#############
def gmm(data, means, classes, stop):
  #Initialize our covariance matrix and prior
  #The centroids are our means
  (covariance, prior) = init_params(data, means, classes)

  #Init loop variables
  i = 0
  delta = stop
  prev_likelihood = 0

  while delta >= stop:
    #Find the log likelihood and print
    likelihood = log_like(data, means, covariance, prior)
    print "Epoch {}:".format(i)
    print "Log-likelihood = {}".format(likelihood)

    #E-step - find membership grades 
    mg = membership_grades(data, means, covariance, prior)
    #Get classification count
    classes = class_count(mg)

    #M-step - find parameters
    means = calc_means(data, mg, classes)
    covariance = calc_cov(data, means, mg, classes, covariance)
    prior = np.true_divide(classes, len(data))

    #Find our delta and check the stopping condition
    delta = likelihood - prev_likelihood
    prev_likelihood = likelihood
    i += 1
  return (mg, means, likelihood)

######################
##--INITIALIZATION--##
######################
#Use the prior classifications to find our initial parameters:
#mean, covariance, and prior
def init_params(data, centroids, classes):
  covariance = []
  prior = []
  classes = classes.flatten()
  for i in range(len(centroids)):
    #condition contains an array of bools that tell whether the point
    #is in the class i or not
    condition = np.mod(classes, len(centroids)) == i
    #pt_list will contain the points for the i-th class only
    pt_list = np.compress(condition, data, axis=0)
    covariance.append(np.cov(pt_list.T))
  
    #Prior is the ratio of points in class i to total points
    prior.append(np.true_divide(len(pt_list), len(classes)))
  prior = np.array(prior)
  return (covariance, prior)

###################
##--CLASS COUNT--##
###################
#Returns a count of classifications per class
def class_count(mg):
  classes = np.zeros((len(mg[0]), 1))
  #Iterate through each set of membership grades and find the max
  for n in range(len(mg)):
    classes[np.argmax(mg[n])] += 1
  return classes

################
##--CLASSIFY--##
################
def gmm_classify(mg):
  classes = np.zeros((len(mg),1))
  for n in range(len(mg)):
    classes[n] = np.argmax(mg[n])
  return classes

####################
##--CALCULATIONS--##
####################
#Calculates the log likelihood based on given parameters
def log_like(data, means, covariance, prior):
  #Create a blank N x k matrix for our pdfs
  pdfs = np.zeros((len(data), len(means)))
  for n in range(len(data)):
    for k in range(len(means)):
      pdfs[n][k] = multivariate_normal.pdf(data[n], mean=means[k], cov=covariance[k])
  return np.sum(np.dot(prior.T, pdfs.T))

#Calculates the Gaussian density
def membership_grades(data, means, covariance, prior):
  mg = np.zeros((len(data), len(means)))
  for n in range(len(data)):
    #Normalization value
    norm = 0
    for k in range(len(means)):
      mg[n][k] = prior[k]*multivariate_normal.pdf(data[n], mean=means[k], cov=covariance[k])
      norm += mg[n][k]
    mg[n] /= norm
  return mg

#Calculates the new mean
def calc_means(data, mg, classes):
  #Create a blank k x 2 array for our new means
  means = np.zeros((len(mg[0]), 2))

  #Add our weighted points to the new mean
  for n in range(len(data)):
    for k in range(len(mg[0])):
      means[k] += mg[n][k]*data[n]
  
  #Normalize with our class count
  for k in range(len(mg[0])):
    means[k] /= classes[k]
  return means

#Calculates the new covariance
def calc_cov(data, means, mg, classes, shape):
  #Create an empty list to store our new covariances
  covariance = np.zeros_like(shape)

  for k in range(len(means)):
    for n in range(len(data)):
      covariance[k] += mg[n][k]*np.outer((data[n]-means[k]),(data[n]-means[k]))
  
  #Normalize with our class count
  for k in range(len(means)):
    covariance[k] /= classes[k]
  
  return covariance

#Used for graphing GMM (as well as fcm)
#colors  - c colors, each with 4 values for RGBA
#weights - weights for each centroid
def get_colors(colors, weights):
  sqrd_colors = np.square(colors)
  sqrd_weights = np.square(weights)
  ret = np.dot(np.transpose(sqrd_colors), np.transpose(weights))
  return np.sqrt(ret/len(weights[0]))
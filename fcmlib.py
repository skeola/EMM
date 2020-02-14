import numpy as np

######################
##--INITIALIZATION--##
######################
#Do preprocessing on the data and assign weights
def init(data, c):
  #Init the weights matrix
  #Matrix is (n x c)
  weights = np.random.random((len(data), c))
  
  return weights

#Calculate the centroids
def get_centroids(data, c, weights, m):
  #Init centroid container
  n_centroids = np.zeros((c, 2))
  d_centroids = np.zeros((c))

  #UPDATE CENTROIDS
  #Get the sum of the weights and weights*x
  for dt in range(len(data)):
    for k in range(c):
      n_centroids[k] += ((weights[dt][k])**m)*data[dt]
      d_centroids[k] += (weights[dt][k])**m

  for k in range(c):
    n_centroids[k] = n_centroids[k]/d_centroids[k]

  return n_centroids  

################
##--LEARNING--##
################
#Iterate through the data until the centers stop moving
def learn(data, c, weights, m):
  #GET CENTROIDS
  centroids = get_centroids(data, c, weights, m)

  #UPDATE WEIGHTS
  for i in range(len(data)):
    for j in range(c):
      total = 0
      for k in range(c):
        total += ((np.linalg.norm(data[i]-centroids[j]))/(np.linalg.norm(data[i]-centroids[k])))**(2/(m-1))
      weights[i][j] = (total)**(-1)

  return (centroids, weights)

  
################
##--CLASSIFY--##
################
def classify(weights):
  #Classification of each point from 0-k
  classification = np.zeros((len(weights),1))
 
  #Iterate through each point
  for i in range(len(weights)):
    #For each data point, find the highest valued weight
    classification[i] = np.argmin(weights[i])
  
  return classification


#############
##--ERROR--##
#############
def get_error(data, weights, centroids):
  err = 0
  for i in range(len(data)):
    for j in range(len(centroids)):
      err += weights[i][j]*np.linalg.norm(data[i]-centroids[j])

  return err

#Run the fuzzy cmeans algorithm on the data for c clusters, n times
#with fuzzifier m. Run the algorithm r times and return the best
#centroids and weights for the run with least error
#sens is the sensitivity of our stopping condition (delta of centroids)
def fcm(data, c, r, m, sens):
  #Epoch Loop
  for i in range(r):
    print("\n\nEPOCH",i+1,":")
    #Get the initial random weights
    weights = init(data, c)
    
    #Run the algorithm once to initialize prev
    (centroids, weights) = learn(data, c, weights, m)
    prev_c = centroids*2
    
    count = 0
    #Repeat training until the centers stop moving
    while np.amax(prev_c - centroids) > sens:
      prev_c = np.copy(centroids)
      (centroids, weights) = learn(data, c, weights, m)
      count += 1

    print("# of iters: ", count)
    err = get_error(data, weights, centroids)
    print("run error:", err)
    #Training is done, compare to previous error minimum
    if i==0:
      (m_wei, m_err) = (weights, err)
    else:
      if err < m_err:
        (m_wei, m_err) = (weights, err)
    print("min error:",m_err)
  #Return results with minimum wcss error
  return (m_wei, m_err)

#This function probably doesn't belong in a fcm library, but I
#wanted to keep it separate from the driver
#colors  - c colors, each with 4 values for RGBA
#weights - weights for each centroid
def get_colors(colors, weights):
  sqrd_colors = np.square(colors)
  sqrd_weights = np.square(weights)
  ret = np.dot(np.transpose(sqrd_colors), np.transpose(weights))
  return np.sqrt(ret/len(weights[0]))

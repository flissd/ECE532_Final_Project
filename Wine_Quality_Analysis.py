import numpy as np
import matplotlib.pyplot as plt
import csv
import math

def recalc_cent(assignments,x,k,centroids):
    cent = np.zeros((k,len(x[0])))
    count = np.zeros(k)
    for j in range(len(assignments)):
        index = int(assignments[j])
        cent[index,:] = cent[index,:] + x[j,:]
        count[index] += 1
    for i in range(k):
        if count[i] > 0:
            cent[i,:] = cent[i,:]/count[i]
        else:
            cent[i,:] = centroids[i,:]
    return cent
def get_random_centers(centroids,x,k,i):
    centroids[i] = x[np.random.random_integers(low=0, high=len(x) - 1), :]
    for l in range(i):
        if (centroids[i] == centroids[l]).all():
            return get_random_centers(centroids, x, k, i)
    return centroids[i]

def kmeans_algorithm(x,k):
    old_centroids = np.zeros((k,len(x[0])))
    centroids = np.zeros((k,len(x[0])))
    for i in range(k):
        centroids[i] = get_random_centers(centroids,x,k,i)
    assignments = np.zeros(len(x))
    ind_error = np.zeros(len(x))
    #loop
    while sum(sum(old_centroids-centroids)) != 0:
        for row_i in range(len(x)):
            row = x[row_i]
            shortest_dist = -1
            for i in range(k):
                distance = (sum((centroids[i].reshape(row.shape) - row)**2))**0.5
                if distance < shortest_dist or shortest_dist < 0:
                    shortest_dist = distance
                    assignments[row_i] = i
                    ind_error[row_i] = distance
        old_centroids = centroids
        centroids = recalc_cent(assignments,x,k,centroids)

    #find distance to different centroids
    #assign to centroid with smallest distance
    #recalculate centroids
    #see if changed
    error = sum(ind_error)
    return assignments,error

#import the data from csv file
data = np.genfromtxt('./winequality-red.csv',delimiter=';')
data = data[1:] #get rid of first row (all 'nan' because they were text titles)

#separate the features and the quality
rd = data[:,len(data[0])-1].reshape((len(data),1)) #quality
rw = data[:,:len(data[0])-1] #features

#split data into sets
indices = np.array([[0,99],[100,199],[200,299],[300,399],[400,499],[500,599],[600,699],[700,799],[800,899],[900,999],[1000,1099],
           [1100,1199],[1200,1299],[1300,1399],[1400,1499],[1500,1598]])
num_i = len(indices)

sq_err = 0
for index in range(num_i):
    start = indices[index,0]
    end = indices[(index+1)%(num_i),1]
    if len(rw[:start]) == 0:
        x_train = rw[end+1:]
        y_train = rd[end+1:]
        x_test = rw[:end+1]
        y_test = rd[:end+1]
    elif len(rw[end:]) == 0:
        x_train = rw[:start]
        y_train = rd[:start]
        x_test = rw[start:]
        y_test = rd[start:]
    elif start > end+1:
        x_train = rw[end+1:start]
        y_train = rd[end+1:start]
        x_test = np.vstack((rw[start:],rw[:end+1]))
        y_test = np.vstack((rd[start:],rd[:end+1]))
    else:
        x_train = np.vstack((rw[:start],rw[end+1:]))
        y_train = np.vstack((rd[:start],rd[end+1:]))
        x_test = rw[start:end+1]
        y_test = rd[start:end+1]

    #use linear regression to find model
    w = np.linalg.inv(np.transpose(x_train)@x_train)@np.transpose(x_train)@y_train

    #record average squared error
    sq_err += sum(pow(x_test@w-y_test,2))/len(y_test)
sq_err = sq_err/num_i

#SVD the data
U,s,VT = np.linalg.svd(rw, full_matrices=False)
#print(s)
#find the condition number
cn = s[0]/s[1]
print(cn)

#kmeans algorithm
k_val = 1
k_error_last = 0
k_error = -1
feature_k = np.zeros(len(rw))

while ((k_error_last-k_error) > 0.01*k_error_last) or k_error < 0 or k_error_last < 0:
    k_error_last = k_error
    feature_k,k_error = kmeans_algorithm(rw,k_val)
    #print(k_val)
    k_val = k_val + 1
    #print(k_error_last)
    #print(k_error)

print(k_val)
print(feature_k)

#redo the linear regression with addition group number
#find the new condition number (if it changes)
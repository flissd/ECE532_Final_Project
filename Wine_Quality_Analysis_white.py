import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import sys
from sklearn import linear_model

np.set_printoptions(threshold=sys.maxsize)

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
    assignments = np.zeros((len(x),k))
    temp = np.zeros(len(x))
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
                    temp[row_i] = i
                    ind_error[row_i] = distance
        old_centroids = centroids
        centroids = recalc_cent(temp,x,k,centroids)

    error = sum(ind_error)
    for i in range(len(x)):
        assignments[i,int(temp[i])] = 1
    return assignments,error

def pgd_l1(x,y,lam=0.05):
    clf = linear_model.Lasso(alpha=lam)
    clf.fit(x,y)
    w = clf.coef_
    return w.reshape((len(w),1))

#import the data from csv file
data = np.genfromtxt('./winequality-white.csv',delimiter=';')
data = data[1:] #get rid of first row (all 'nan' because they were text titles)

#separate the features and the quality
rd = data[:,len(data[0])-1].reshape((len(data),1)) #quality
rw = data[:,:len(data[0])-1] #features

#split data into sets
indices = np.array([[0,305],[306,611],[612,917],[918,1223],[1224,1529],[1530,1835],[1836,2141],[2142,2447],
                    [2447,2753],[2754,3059],[3060,3365],[3366,3671],[3672,3977],[3978,4283],[4284,4589],
                    [4589,4897]])
num_i = len(indices)


sq_err = 0
sq_err_lasso = 0
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
    w_ls = np.linalg.inv(np.transpose(x_train)@x_train)@np.transpose(x_train)@y_train
    w_lasso = pgd_l1(x_train,y_train)

    #record average squared error
    sq_err += sum(pow(x_test@w_ls-y_test,2))/len(y_test)
    sq_err_lasso += sum(pow(x_test@w_lasso-y_test,2))/len(y_test)
sq_err = sq_err/num_i
sq_err_lasso = sq_err_lasso/num_i
print("Squared Error: ",sq_err)
print("Least-Squares Weights: \n",w_ls)
print("LASSO Weights: \n",w_lasso)
#SVD the data
U,s,VT = np.linalg.svd(rw, full_matrices=False)
#print(s)
#find the condition number
cn = s[0]/s[1]
print("Condition Number: ",cn)

#kmeans algorithm
k_val = 1
k_error_last = 0
k_error = -1
feature_k = np.zeros(len(rw))

while ((k_error_last-k_error) > 0.1*k_error_last) or k_error < 0 or k_error_last < 0:
    k_error_last = k_error
    feature_k,k_error = kmeans_algorithm(rw,k_val)
    #print(k_val)
    k_val = k_val + 1
    #print(k_error_last)
    #print(k_error)

#print(k_val)
#print(feature_k)

#redo the linear regression with addition group number
rw_add = np.column_stack((rw,feature_k))


new_sq_err = 0
new_sq_err_lasso = 0
for index in range(num_i):
    start = indices[index,0]
    end = indices[(index+1)%(num_i),1]
    if len(rw[:start]) == 0:
        x_train = rw_add[end+1:]
        y_train = rd[end+1:]
        x_test = rw_add[:end+1]
        y_test = rd[:end+1]
    elif len(rw[end:]) == 0:
        x_train = rw_add[:start]
        y_train = rd[:start]
        x_test = rw_add[start:]
        y_test = rd[start:]
    elif start > end+1:
        x_train = rw_add[end+1:start]
        y_train = rd[end+1:start]
        x_test = np.vstack((rw_add[start:],rw_add[:end+1]))
        y_test = np.vstack((rd[start:],rd[:end+1]))
    else:
        x_train = np.vstack((rw_add[:start],rw_add[end+1:]))
        y_train = np.vstack((rd[:start],rd[end+1:]))
        x_test = rw_add[start:end+1]
        y_test = rd[start:end+1]

    #use linear regression to find model
    w_ls_new = np.linalg.inv(np.transpose(x_train)@x_train)@np.transpose(x_train)@y_train
    w_lasso_new = pgd_l1(x_train,y_train)

    #record average squared error
    new_sq_err += sum(pow(x_test@w_ls_new-y_test,2))/len(y_test)
    new_sq_err_lasso += sum(pow(x_test @ w_lasso_new - y_test, 2)) / len(y_test)
new_sq_err = new_sq_err/num_i
new_sq_err_lasso = new_sq_err_lasso/num_i

print("New Squared Error: ",new_sq_err)
print("New Least-Squares Weights: \n",w_ls_new)
print("New LASSO Weights: \n",w_lasso_new)

#SVD the data
U,s,VT = np.linalg.svd(rw_add, full_matrices=False)
#print(s)
#find the condition number
cn = s[0]/s[1]
print("New Condition Number: ",cn)

labels = ['Least-Squares Linear Regression', 'LASSO Regression']
without_features = [sq_err[0], sq_err_lasso[0]]
with_features = [new_sq_err[0], new_sq_err_lasso[0]]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, without_features, width, label='Before')
rects2 = ax.bar(x + width/2, with_features, width, label='After')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Squared Error')
ax.set_title('Average Squared Errors')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

next_labels = ["1","2","3","4",
               "5","6","7","8","9","10","11"]
x2 = np.arange(len(next_labels))
value = w_lasso.T[0]
plt.bar(x2,value,0.35)
plt.title("Weights Without K-Means")
plt.ylabel("Weight")
plt.xlabel("Feature")
plt.show()


x2 = np.arange(len(w_lasso_new.T[0]))
value = w_lasso_new.T[0]
plt.bar(x2,value,0.35)
plt.title("Weights With K-Means")
plt.ylabel("Weight")
plt.xlabel("Feature")
plt.show()
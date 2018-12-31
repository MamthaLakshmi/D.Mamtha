import math
import numpy as np
import pandas as pd

#data/Student_Performance
df = pd.read_csv("stud.csv")

# fetch all features into different lists

##id = df['test preparation course'].values
f1 = df['math score'].values
f2 = df['reading score'].values
f3 = df['writing score'].values
##f4 = df['test preparation course'].values


math_score= df['math score'].values
reading_score= df['reading score'].values
writing_score= df['writing score'].values
test_preparation_course= df['test preparation course'].values
test_completed=[]
test_notcompleted=[]
f1=[]
f2=[]
f3=[]
f11=[]
f12=[]
f13=[]
##print test_preparation_course
for i in range(len(test_preparation_course)):
    if(test_preparation_course[i]=="completed"):
        f1.append(math_score[i])
        f2.append(reading_score[i])
        f3.append(writing_score[i])
    else:
        f11.append(math_score[i])
        f12.append(reading_score[i])
        f13.append(writing_score[i])
        
# Storing original data into an array
original_data = np.array(list(zip(f1,f2,f3)))
original_data1 = np.array(list(zip(f11,f12,f13)))

# data array with only the features which will be used to cluster the data
data = np.array(list(zip(f1,f2,f3)))
data1 = np.array(list(zip(f11,f12,f13)))


def init_centroids(k, data):
    
    c = []
    s = np.random.randint(low=1, high=len(data), size=k)
   
    while (len(s) != len(set(s))):
        s = np.random.randint(low=1, high=len(data), size=k)
    
    for i in s:
        c.append(data[i])
    return c

def euc_dist(a, b):
    
    sum = 0
    for i, j in zip(a, b):
        a = (i - j) * (i - j)
        sum = sum + a
    return math.sqrt(sum)

def cal_dist(centroids, data):
   
    c_dist = []
    # For each centroid c, iterate through all points in data to calculate its distance from c
    for i in centroids:
        temp = []
        for j in data:
            temp.append(euc_dist(i, j))
        c_dist.append(temp)
    return c_dist

def perf_clustering(k, dist_table):
    
    # create empty cluster list of size k
    clusters = []
    for i in range(k):
        clusters.append([])
    # start clustering data points, such that each point is clustered to nearest centroid
    for i in range(len(dist_table[0])):
        d = []
        for j in range(len(dist_table)):
            d.append(dist_table[j][i])
        clusters[d.index(min(d))].append(i)
    return clusters

def update_centroids(centroids, cluster_table, data):
    
    for i in range(len(centroids)):
        # Update the centroid if there are some flowers within this centroid
        if (len(cluster_table[i]) > 0):
            temp = []
            # Copy features of cluster members to temp list
            for j in cluster_table[i]:
                temp.append(list(data[j]))
            # Take mean of features of all members of cluster to get new centroid location
            sum = [0] * len(centroids[i])
            for l in temp:
                sum = [(a + b) for a, b in zip(sum, l)]
            centroids[i] = [p / len(temp) for p in sum]

    return centroids


def check_n_stop(dist_mem, cluster_mem):
    
    # Check if distance table has not changed over past iterations
    c1 = all(x == dist_mem[0] for x in dist_mem)
    # Check if cluster table has not changed over past iterations
    c2 = all(y == cluster_mem[0] for y in cluster_mem)

    if c1:
        print("Stopping... Distance table has not changed from few iterations")
    elif c2:
        print("Stopping... Cluster table has not changed from few iterations")
    return c1 or c2


def kMeans(k, data, max_iterations):
    

    # These lists will maintain memory to check if stopping criteria is met
    dist_mem = []
    cluster_mem = []

    # Initialize centroids
    centroids = init_centroids(k, data)
    # Calculate distance table
    distance_table = cal_dist(centroids, data)
    # Perform clustering based on above generated distance table
    cluster_table = perf_clustering(k, distance_table)
    # Update centroid location based on above generated cluster table
    newCentroids = update_centroids(centroids, cluster_table, data)

    # Add distance and cluster table to memory list
    dist_mem.append(distance_table)
    cluster_mem.append(cluster_table)

    # Repeat from step 2 till stopping criteria is met
    for i in range(max_iterations):
        distance_table = cal_dist(newCentroids, data)
        cluster_table = perf_clustering(k, distance_table)
        newCentroids = update_centroids(newCentroids, cluster_table, data)

        dist_mem.append(distance_table)
        cluster_mem.append(cluster_table)
        # If distance/cluster has not changed over last 10 iterations, stop, else continue
        if len(dist_mem) > 10:
            dist_mem.pop(0)
            cluster_mem.pop(0)
            if check_n_stop(dist_mem, cluster_mem):
                print("Stopped at iteration #", i)
                break

    # Display the final results
    for i in range(len(newCentroids)):
        print("Centroid #", i, ": ", newCentroids[i])
        print("Members of the cluster: ")
        for j in range(len(cluster_table[i])):
            a=original_data[cluster_table[i][j]]
            test_completed.append(a)
            print(original_data[cluster_table[i][j]])
##            print(original_data1[cluster_table[i][j]])
            
def kMeans1(k, data, max_iterations):
    

    # These lists will maintain memory to check if stopping criteria is met
    dist_mem = []
    cluster_mem = []

    # Initialize centroids
    centroids = init_centroids(k, data)
    # Calculate distance table
    distance_table = cal_dist(centroids, data)
    # Perform clustering based on above generated distance table
    cluster_table = perf_clustering(k, distance_table)
    # Update centroid location based on above generated cluster table
    newCentroids = update_centroids(centroids, cluster_table, data)

    # Add distance and cluster table to memory list
    dist_mem.append(distance_table)
    cluster_mem.append(cluster_table)

    # Repeat from step 2 till stopping criteria is met
    for i in range(max_iterations):
        distance_table = cal_dist(newCentroids, data)
        cluster_table = perf_clustering(k, distance_table)
        newCentroids = update_centroids(newCentroids, cluster_table, data)

        dist_mem.append(distance_table)
        cluster_mem.append(cluster_table)
        # If distance/cluster has not changed over last 10 iterations, stop, else continue
        if len(dist_mem) > 10:
            dist_mem.pop(0)
            cluster_mem.pop(0)
            if check_n_stop(dist_mem, cluster_mem):
                print("Stopped at iteration #", i)
                break

    # Display the final results
    for i in range(len(newCentroids)):
        print("Centroid #", i, ": ", newCentroids[i])
        print("Members of the cluster: ")
        for j in range(len(cluster_table[i])):
##            print(original_data[cluster_table[i][j]])
            a=original_data1[cluster_table[i][j]]
            test_notcompleted.append(a)
            print(original_data1[cluster_table[i][j]])


# Run the K-Means algorithm on the Iris-Dataset with k = 3, and max-iterations limited to 100
##print data
kMeans(1, data, 1001)
kMeans1(1, data1, 1001)
print "test completed students:",len(test_completed)
print "test not completed students:",len(test_notcompleted)




#scatter_plot
import matplotlib.pyplot as plt
import csv
math_score1=[]
read_score1=[]
math_score2=[]
read_score2=[]
with open('StudentsPerformance.csv','r') as csvfile:
    plots=csv.reader(csvfile,delimiter=',')
    for column in plots:
        a=str(column[4])
        if(a=="completed"):
            math_score1.append(int(column[5]))
            read_score1.append(int(column[6]))
        else:
            math_score2.append(int(column[5]))
            read_score2.append(int(column[6]))
            
            

figure=plt.figure()
plt.axis([0,120,0,120])
line1=plt.scatter(math_score1,read_score1,color='g')
line2=plt.scatter(math_score2,read_score2,color='r')
##plt.plot(math_score,read_score,color="blue",markeredgecolor="black")
plt.title("StudentsPerformance")
plt.xlabel("math score")
plt.ylabel("reading score")
plt.legend((line1, line2), ('Completed', 'Not Completed'))
plt.show()
figure.savefig("scatterplot.svg")




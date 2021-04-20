#19CS10031 | Gandhi Abhishek Rajesh
#CS1
#Cricket Format Clustering using Single Linkage Hierarchical Clustering Technique

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random # for genrationg random number, random seed 8,12 works fantastically 
import time

start = time.time()
nth = { #used in printing
    0:"zeroth",
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth"
}

random.seed(8) #Random seed 5,8,12 works best mostly because using these values our code runs for at max 20 iteration
#and if we use these values our code converges before 20 iteration
class bucket: # store unit fo k_means and all usefull functions needed in implementing the code
    def __init__(self,mean):
        self.mean = mean
        self.members = []
        self.member_index = []
        
    def update_mean(self,mean): 
        if (self.mean == mean).all()==True:
            return True
        self.mean = mean
        return False

    def add_member(self,member,i):
        self.members.append(member)
        self.member_index.append(i)

    def return_array_mean(self):
        if(len(self.members)==0):
            print("zero members")
            return self.mean
        return sum(self.members)/len(self.members)

    def get_distance(self,input):
        return np.linalg.norm(input-self.mean)

    def reset_list(self):
        self.members = []
        self.member_index = []

class k_means: #k_means class 
    def __init__(self,k):
        self.k=k
        self.bucket = []
        self.max_iter = 20

    def distance(self,input1,input2):
        return np.linalg.norm(input1-input2)

    def init_bucket(self,data): #initialized all the containers using value of k and provided data
        n,d = data.shape
        for i in range(self.k):
            self.bucket.append(bucket(data[random.randrange(0,n)]))

    def closest_bucket(self,input): 
        index = 0
        min = self.bucket[0].get_distance(input)
        for i in range(1,self.k):
            if min > self.bucket[i].get_distance(input):
                index = i
                min = self.bucket[i].get_distance(input)
        return index

    def fit(self,data): #perform interations till a optimum is achieved or 20 iterations are done
        n,d = data.shape
        self.init_bucket(data)
        unchanged = False
        count=0
        while(not unchanged and count < self.max_iter ):
            count+=1
            unchanged = True
            for i in range(self.k):
                self.bucket[i].reset_list()
            for i in range(n):
                index = self.closest_bucket(data[i])
                self.bucket[index].add_member(data[i],i)
            for i in range(self.k):
                new_mean = self.bucket[i].return_array_mean()
                if not self.bucket[i].update_mean(new_mean):
                    unchanged = False
                    
    def compute_silhouette(self): #computes silhoette cofficient by individually calculating a and b for every point
        list_bucket = []
        value_bucket = 0
        for i in range(self.k):
            si=[]
            for j in self.bucket[i].members:
                ai=0
                for k in self.bucket[i].members:
                    ai+=self.distance(j,k)
                ai=(ai)/(len(self.bucket[i].members)-1)
                b_values = []
                for k in range(self.k):
                    if k!=i:
                        temp=0
                        for m in self.bucket[k].members:
                            temp+=self.distance(j,m)
                        b_values.append(temp/(len(self.bucket[k].members)))
                bi=min(b_values)
                si.append((bi-ai)/max(ai,bi))
            list_bucket.append(sum(si)/len(si))
        return max(list_bucket)

class storage: #used in Hierarchical_Clustering as a storage unit, contains all the required functions
    def __init__(self):
        self.members = []
        self.member_index = []

    def add_members(self,member,i):
        self.members.append(member)
        self.member_index.append(i)
        
    def merge(self,store): #for merging two such objects
        s = storage()
        s.members = self.members+store.members
        s.member_index = self.member_index + store.member_index
        return s

    def distance(self,input1,input2): #euclidian distance
        return np.linalg.norm(input1-input2)

    def find_min_distance(self,storage):
        min = float('inf')
        for i in self.members:
            for j in storage.members:
                temp = self.distance(i,j)
                if(temp<min):
                    min=temp
        return min
    
class Hierarchical_Clustering: #performs single linkage Hierarchical_Clustering
    def __init__(self,k):
        self.storage_set = []
        self.k = k
        self.distance_matrix = []
        self.valid_locations = []
        self.final_sets = []

    def distance(self,input1,input2): #euclidian distance
        return np.linalg.norm(input1-input2)

    def initialize_storage(self,data): # intialized storage container and 2-D distance matrix
        n,d = data.shape
        for i in range(n):
            self.storage_set.append(storage())
            self.storage_set[i].add_members(data[i],i)
            temp = []
            for j in range(n):
                temp.append(float('inf'))
            self.distance_matrix.append(temp)
            self.valid_locations.append(1)
        
        for i in range(len(self.storage_set)):
                for j in range(i):
                    if(j!=i):
                        temp = self.storage_set[i].find_min_distance(self.storage_set[j])
                        self.distance_matrix[i][j] = temp
        self.distance_matrix = np.array(self.distance_matrix)

    def fit(self,data): #uses a dp approch same as Kruskal's algorithm updating distance matrix after every step
        self.initialize_storage(data) #untill we reaches our optimum solution, very fast O(n^2)
        n,d = data.shape
        count = n
        while(np.sum(self.valid_locations)>self.k):
            result = np.unravel_index(np.argmin(self.distance_matrix, axis=None), self.distance_matrix.shape)
            merged_set = self.storage_set[result[0]].merge(self.storage_set[result[1]])
            self.valid_locations[result[1]]=0
            self.storage_set[result[0]] = merged_set
            for i in range(n):
                temp = np.array([self.distance_matrix[result[0],i],self.distance_matrix[result[1],i],                               self.distance_matrix[i,result[0]],self.distance_matrix[i,result[1]]])
                if(i<result[0] and i!=result[1]):
                    self.distance_matrix[result[0],i] = np.min(temp)
                if(i>result[0] and i!=result[1]):
                    self.distance_matrix[i,result[0]] = np.min(temp)
                self.distance_matrix[i,result[1]] = float('inf')
                self.distance_matrix[result[1],i] = float('inf')
            count-=1
        for i in range(n):
            if(self.valid_locations[i]==1):
                self.final_sets.append(self.storage_set[i])


    def fit_hard(self,data): #hard coded computing all the distances at every point O(n^3) #never called in final submission, used for verification 
        self.initialize_storage(data)    #very slow 
        n,d = data.shape
        count = n
        while(count>self.k):
            min = float('inf')
            set1 = None
            set2 = None
            for i in self.storage_set:
                for j in self.storage_set:
                    if(j!=i):
                        temp = i.find_min_distance(j)
                        if(temp<min):
                            set2= i
                            set1= j
                            min = temp
            self.storage_set.remove(set1)
            self.storage_set.remove(set2)
            self.storage_set.append(set1.merge(set2))
            count-=1
        self.final_sets = self.storage_set

    def compute_silhouette(self): #computes silhouette cofficient
        list_bucket = []
        value_bucket = 0
        for i in range(self.k):
            si=[]
            for j in self.final_sets[i].members:
                ai=0
                for k in self.final_sets[i].members:
                    ai+=self.distance(j,k)
                ai=(ai)/(len(self.final_sets[i].members)-1)
                b_values = []
                for k in range(self.k):
                    if k!=i:
                        temp=0
                        for m in self.final_sets[k].members:
                            temp+=self.distance(j,m)
                        b_values.append(temp/(len(self.final_sets[k].members)))
                bi=min(b_values)
                si.append((bi-ai)/max(ai,bi))
            list_bucket.append(sum(si)/len(si))
        return max(list_bucket)

class Jagard_similarity: #takes in a k_mean class and a Hierarchical_Clustering class and calculates jagard similarity
    def __init__(self,k_means,Hierarchical_Clustering):
        self.kmeans = k_means
        self.Hier_cluster = Hierarchical_Clustering
    def intersection(self,lst1, lst2): 
        lst3 = [value for value in lst1 if value in lst2] 
        return lst3
    def Union(self,lst1, lst2): 
        final_list = list(set(lst1) | set(lst2)) 
        return final_list 
    def jagard_simple(self,list1,list2):
        return len(self.intersection(list1,list2))/len(self.Union(list1,list2))
    def compute(self): #calculates jagard similarity (intersection/union)
        final_values = []
        temp2 =[]
        for i in self.kmeans.bucket:
            temp2.append(min(i.member_index))
        for i in self.kmeans.bucket:
            temp = []
            for j in self.Hier_cluster.final_sets:
                temp.append(self.jagard_simple(self.kmeans.bucket[temp2.index(min(temp2))].member_index,j.member_index))
            final_values.append(max(temp))
            temp2[temp2.index(min(temp2))] = float('inf')
        return final_values


class save_to_file: #create and save in required format
    def save_k_means(self,k_mean):
        f = open("kmeans.txt","w")
        index_list = []
        list_order = []
        for i in k_mean.bucket:
            i.member_index.sort()
            index_list.append(i.member_index)
            list_order.append(i.member_index[0])
        for i in range(k_mean.k):
            minpos = list_order.index(min(list_order))
            list_order[minpos] = float('inf')
            s = ""
            for i in index_list[minpos]:
                s+=str(i)+","
            s = s[:-1]
            f.write(s)
            f.write('\n')

    def save_hier(self,hier):
        f = open("agglomerative.txt","w")
        index_list = []
        list_order = []
        for i in hier.final_sets:
            i.member_index.sort()
            index_list.append(i.member_index)
            list_order.append(i.member_index[0])
        for i in range(hier.k):
            minpos = list_order.index(min(list_order))
            list_order[minpos] = float('inf')
            s = ""
            for i in index_list[minpos]:
                s+=str(i)+","
            s = s[:-1]
            f.write(s)
            f.write('\n')

def main(): #calls everything systamatically and prints computed values
    sample = pd.read_csv("cricket_1_unlabelled.csv")
    train = np.array(sample)
    train = train[:,1:] #Removing index list
    print("Input size is",train.shape)
    train = (train-np.sum(train,axis=0)/train.shape[0])/np.std(train,axis=0) #peforming normalisation for better result
    best_k = 3
    best_k_mean = None
    best_value = -1
    print("++++++++++++++++++++++++++++Performing k-mean Algorithm+++++++++++++++++++++++++++++++++++++++++++")
    for k in range(3,7):
        k_mean = k_means(k)
        k_mean.fit(train.astype(np.float64))
        temp = k_mean.compute_silhouette()
        if(best_value<temp):
            best_value = temp
            best_k = k
            best_k_mean = k_mean #storing class with best k value
        print("Silhouette Distance calculated for value of k =",k,"is",temp)
    print("Best value of k is",best_k_mean.k)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    print("++++++++++++++++++++++++++++Performing hierarchical clustering algorithm++++++++++++++++++++++++++")
    h = Hierarchical_Clustering(best_k)
    h.fit(train.astype(np.float64)) 
    print("Using value of k =",best_k)
    print("silhouette cofficient for single linkage is",h.compute_silhouette())
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    s = save_to_file()
    s.save_k_means(best_k_mean)
    s.save_hier(h)
    print("++++++++++++++++++++++++++++++++Calculating Jagard_similarity+++++++++++++++++++++++++++++++++++++")
    j= Jagard_similarity(best_k_mean,h)
    scores = j.compute()
    for i in range(len(scores)):
        print("Jagard_similarity for",nth[i],"cluster is",scores[i])
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

if __name__=='__main__':
    main()    

end = time.time()
print("Time Taken for complete code running is",(end-start),"seconds")
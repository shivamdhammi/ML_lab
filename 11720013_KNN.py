import numpy as np
import pandas as pd
import csv
from math import sqrt

data = []
with open('knn_algo.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    # print(reader)
    for row in reader:
        data.append (row)

# print(data)
attributes = data[0]
data = data[1:]
# print("Attributes is:", attributes)

k = input("Enter Value of K for the algo : ")
k = int(k)

print("Enter the details about the Person you want to guess the sports output")
name = input("Name : ")
age = input("Class : ")
gender = input("Gender(M/F) : ")
if gender == 'M':
    gender = 0
else:
    gender = 1

test_row = [name, int(age), gender]

def transform_data(data):
    new_data = []
    for row in data:
        p = row
        p[1] = int(p[1])
        if p[2] == 'Male':
            p[2] = 0
        else:
            p[2] = 1
        new_data.append(p)
    return new_data

def euclidean_dist(train_row, test_row):
    distance = 0.0
    size = len(test_row)
    for i in range(1,size):
        distance += (train_row[i] - test_row[i])**2
    return sqrt(distance)

def find_distances(train_data, test_row):
    dist_list = list()
    for train_row in train_data:
        dist = euclidean_dist(train_row, test_row)
        dist_list.append((train_row, dist))
    dist_list.sort(key=lambda tup:tup[1])
    return dist_list

def k_nearest_neighbour(dist_list, num_neigbours):
    neighbours = list()
    for i in range(num_neigbours):
        neighbours.append(dist_list[i][0])
    return neighbours

def predict_class(neighbour_data):
    output_values = [row[-1] for row in neighbour_data]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

processed_data = transform_data(data)
# print(processed_data)

dist_list = find_distances(processed_data, test_row)
# print(dist_list)

neighbour_data = k_nearest_neighbour(dist_list, k)
# print(neighbour_data)

predicted_data = predict_class(neighbour_data)
print("Predicted Class is : ", predicted_data)
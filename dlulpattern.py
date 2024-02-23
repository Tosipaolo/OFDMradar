import numpy as np
from find_num_diff_resampling import *

vector = []


for i in range(8):
    for j in range(140):
        if j < 104: vector.append(1)
        else: vector.append(0)

pace = 0
shift = 0

sampled_vector = [1, 0]

while( any(item == 0 for item in sampled_vector) ):

    pace = pace + 1

    sampled_vector = []

    num_samples = len(vector) // pace

    for i in range(num_samples):
        sampled_vector.append(vector[i * pace])

num_diff_samples = find_num_diff_resampling(vector, pace)



print(f"minimum pace is {pace}")
print(sampled_vector)
index_list = []
print(num_samples)
for num in range(num_samples):
    index_list.append(num)

print(np.multiply(pace, index_list))
print(f"Number of different resampling at pace {pace}: {num_diff_samples}")
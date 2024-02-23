import numpy as np
def find_num_diff_resampling(vector, dec_pace):
    shift = -1
    sampled_vect = [1]
    num_samples = len(vector) // dec_pace

    while(all(item == 1 for item in sampled_vect)):
        sampled_vect = []

        shift +=1
        for i in range(num_samples):
            sampled_vect.append(vector[(i * dec_pace) + shift])
        print(f"shift is: {shift}")

    return shift - 1
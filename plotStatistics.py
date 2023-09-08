import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows as windows
import scipy as sp

def plot_frame_pdf(frame):

    histogram = np.histogram(frame, bins=150)

    # constant bins width, density can be ignored
    dist = sp.stats.rv_histogram(histogram, density=False)

    fig = plt.figure()
    plt.stairs(histogram)

    return 1
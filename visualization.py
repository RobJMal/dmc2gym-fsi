import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def save_video(frames, filename='media/output.mp4', framerate=30):
    '''
    Creates a video of the provided frames 
    '''
    fig = plt.figure(figsize=(frames[0].shape[1] / 100, frames[0].shape[0] / 100), dpi=100)
    plt.axis('off')

    print("Creating video...")
    ims = [[plt.imshow(frame, animated=True)] for frame in frames]
    ani = animation.ArtistAnimation(fig, ims, interval=1000/framerate, blit=True, repeat_delay=1000)
    ani.save(filename, writer='ffmpeg', fps=framerate)

    print("Finished creating video")
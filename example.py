import os 
import copy 
import itertools
from IPython.display import clear_output

# Sets the graphics library for MuJoCo, need to set this 
# before importing dm_control 
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'    # Connecting to GPU

# Main imports 
import dmc2gym_fsi 
import matplotlib.pyplot as plt
import numpy as np 

# Custom libraries 
from visualization import save_video

if __name__ == '__main__':

    # Ensure the media directory exists
    os.makedirs("media", exist_ok=True)

    random_state = np.random.RandomState(42)    # Setting the seed 

    # Variables from fish environment 
    domain_name = "fish"
    task_name = "swim"
    seed = 42
    visualize_reward = False
    from_pixels = True
    height = 64 # Shape of a single image in the frame 
    width = 64  # Shape of a single image in the frame
    frame_skip = 2
    pixel_norm = True

    env = dmc2gym_fsi.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=visualize_reward,
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
    )
    pixels = env.physics.render()

    duration = 10
    frames = []
    ticks = []
    rewards = []
    observations = []

    spec = env.action_spec()    # Environment specifications (range and shape of valid actions)
    time_step = env.reset()

    while env.physics.data.time < duration: 
        action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
        obs, reward, done, extra = env.step(action)

        frames.append(obs)
        rewards.append(reward)
        observations.append(copy.deepcopy(obs))
        ticks.append(env.physics.data.time)

    output_policy_video_filename = f'media/{domain_name}-{task_name}_result.mp4'
    save_video(frames=frames, filename=output_policy_video_filename, framerate=30)
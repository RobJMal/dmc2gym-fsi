import os 
import copy 
import itertools
from IPython.display import clear_output

# Sets the graphics library for MuJoCo, need to set this 
# before importing dm_control 
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'    # Connecting to GPU

# Main imports 
import dmc2gym
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image

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
    height = 64
    width = 64
    frame_skip = 2
    pixel_norm = True

    env = dmc2gym.make(
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

    duration = 4
    frames = []
    ticks = []
    rewards = []
    observations = []

    spec = env.action_spec()    # Environment specifications (range and shape of valid actions)
    time_step = env.reset()

    while env.physics.data.time < duration: 
        action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
        obs, reward, done, extra = env.step(action)

        obs = np.transpose(obs, (1, 2, 0))
        obs_image = Image.fromarray(obs)
        obs_image.save("test1.png")
        breakpoint()

        # Frames of the agent in the environment (for visualization purposes)
        # camera0 = env.physics.render(camera_id=0, height=200, width=200)
        # camera1 = env.physics.render(camera_id=1, height=200, width=200)
        frames.append(np.hstack((camera0, camera1)))
        # breakpoint()
        rewards.append(reward)
        observations.append(copy.deepcopy(obs))
        ticks.append(env.physics.data.time)

    output_policy_video_filename = f'media/{domain_name}-{task_name}_result.mp4'
    save_video(frames=frames, filename=output_policy_video_filename, framerate=30)
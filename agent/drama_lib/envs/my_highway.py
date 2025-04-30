import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

class HighwayEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        name,
        size=(64, 64),
        vehicle_count=20,
        length=108000,
        scaling=1.75,
        polcicy_frequency=2,
        seed=None,
    ):
        if name.startswith("HW"):
            name = name[3:]
        self._env_id = name
        self._size = size
        self._gray = True
        self._length = length
        self._random = np.random.RandomState(seed)
        self._config = {
        "observation": {
            "type": "GrayscaleObservation",       
            "observation_shape": size,
            "stack_size": 1,
            "vehicles_count": vehicle_count,
            # 'screen_height': size[0],
            # 'screen_width': size[1],
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": scaling,
        },
        "action": {
            "type": "DiscreteMetaAction"
        },        
        "policy_frequency": polcicy_frequency
        }
        self._env = gym.make(self._env_id, config=self._config, render_mode="rgb_array")
        shape = self._env.observation_space.shape
        self._buffer = [np.zeros(shape, np.uint8) for _ in range(2)]
        self._seed = seed


    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Box(0, 255, img_shape, np.uint8)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        terminated = truncated = False
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._buffer[0] = obs
        self._step += 1
        my_truncated = self._length and self._step >= self._length
        self._done = terminated or my_truncated or truncated
        return self._obs(reward, info, is_last=self._done, is_terminal=terminated)

    def reset(self):
        obs, info = self._env.reset()
        self._buffer[0] = obs
        self._done = False
        self._step = 0
        self._episode_frame_num = 0
        
        obs, reward, is_last, info = self._obs(0.0, info, is_first=True)
        return obs, info

    def _grayscale_to_colormap(self, image, cmap='viridis'):   
        """
        Converts a grayscale image (shape: 1, H, W) to an RGB image using a Matplotlib colormap.
        Output shape: (3, H, W), dtype: uint8.
        """
        if image.shape[0] != 1:
            raise ValueError("Input image must have shape (1, height, width)")
        
        # Squeeze to 2D (H, W) and normalize to [0, 1]
        grayscale = image.squeeze(0).astype(np.float32) / 255.0
        
        # Get the colormap function
        colormap = plt.get_cmap(cmap)
        
        # Apply colormap: (H, W) → (H, W, 4) RGBA
        rgba_image = colormap(grayscale)
        
        # Convert to RGB (discard alpha), scale to [0, 255], and transpose to (3, H, W)
        rgb_image = (rgba_image[..., :3] * 255).astype(np.uint8)
        # rgb_image = np.transpose(rgb_image, (2, 0, 1))  # Change axis order to (3, H, W)
        
        return rgb_image

    def _obs(self, reward, info, is_first=False, is_last=False, is_terminal=False):
        image = self._buffer[0]
        # # Process the observation
        if self._gray:
            image = self._grayscale_to_colormap(image)
        info.update({
            'is_first': is_first,
            'is_terminal': is_terminal,
            'episode_frame_number': self._step,
        })
        return (image, reward, is_last, info)

    def close(self):
        return self._env.close()
    
if __name__ == "__main__":
    env = HighwayEnv('HW/highway-fast-v0')
    obs, info = env.reset()

    # # Move channels to the last dimension: (3, 64, 64) -> (64, 64, 3)
    # # obs = np.transpose(obs, (1, 2, 0))

    # plt.imshow(obs)
    # plt.axis('off')  # Optional: Turn off axis labels
    # plt.show()

    # next_obs, reward, terminated, info = env.step(0)

    #     # Move channels to the last dimension: (3, 64, 64) -> (64, 64, 3)
    # # next_obs = np.transpose(next_obs, (1, 2, 0))

    # plt.imshow(next_obs)
    # plt.axis('off')  # Optional: Turn off axis labels
    # plt.show()   

    import time
    
    # Start the timer
    start_time = time.time()

    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs, info = env.reset()
    
    # End the timer
    end_time = time.time()
    env.close()
    # Print the elapsed time
    print(f"Time taken for 1,000 iterations: {end_time - start_time:.2f} seconds")
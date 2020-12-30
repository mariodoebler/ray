import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
from PIL import Image
from atariari.benchmark.ram_annotations import atari_dict
from pathlib import Path
import os
import torch

def is_atari(env):
    if (hasattr(env.observation_space, "shape")
            and env.observation_space.shape is not None
            and len(env.observation_space.shape) <= 2):
        return False
    return hasattr(env, "unwrapped") and hasattr(env.unwrapped, "ale")


def get_wrapper_by_cls(env, cls):
    """Returns the gym env wrapper of the given class, or None."""
    currentenv = env
    while True:
        if isinstance(currentenv, cls):
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            return None


class MonitorEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Record episodes stats prior to EpisodicLifeEnv, etc."""
        gym.Wrapper.__init__(self, env)
        self._current_reward = None
        self._num_steps = None
        self._total_steps = None
        self._episode_rewards = []
        self._episode_lengths = []
        self._num_episodes = 0
        self._num_returned = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)

        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1

        self._current_reward = 0
        self._num_steps = 0

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        return (obs, rew, done, info)

    def get_episode_rewards(self):
        return self._episode_rewards

    def get_episode_lengths(self):
        return self._episode_lengths

    def get_total_steps(self):
        return self._total_steps

    def next_episode_results(self):
        for i in range(self._num_returned, len(self._episode_rewards)):
            yield (self._episode_rewards[i], self._episode_lengths[i])
        self._num_returned = len(self._episode_rewards)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """ Do no-op action for a number of             obs_ram = np.array([obs[self.ram_variables_dict["ball_x"]],
                                obs[self.ram_variables_dict["enememy_y"]],
                                obs[self.ram_variables_dict["player_y"]],
                                obs[self.ram_variables_dict["ball_y"]]], dtype='float64')
steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset.

        For environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2, ) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class WarpRectangularFrame(gym.ObservationWrapper):
    def __init__(self, env, dim_height, dim_width):
        """Warp frames to the specified size (dim_height x dim_width)."""
        gym.ObservationWrapper.__init__(self, env)
        self.height = dim_height
        self.width = dim_width
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, dim):
        """Warp frames to the specified size (dim x dim)."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

def wrap_deepmind(env, dim=84, framestack=True):
    """Configure environment for DeepMind-style Atari.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
    """
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if "NoFrameskip" in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, dim)
    # env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    if framestack:
        env = FrameStack(env, 4)
    return env

def wrap_rectangular_deepmind(env, dim_height=210, dim_width=160, framestack=False):
    """Configure environment for DeepMind-style Atari.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
    """
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if "NoFrameskip" in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpRectangularFrame(env, dim_height=dim_height, dim_width=dim_width)
    # env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    if framestack:
        env = FrameStack(env, 4)
    return env
    
def wrap_ram(env, framestack=True, extract_ram=True, debug_trajectory=False):
    # ORDER is important
    # FIRST extract rams, then (maybe) stack the observations
    # env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    # if "NoFrameskip" in env.spec.id:
        # env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # env = WarpFrame(env, dim)
    # env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval


    if extract_ram:
        env = ExtractRAMLocations(env, debug_trajectory)

    if framestack:
        env = FrameStackRAMFrameSkip(env, k=4, skip=4)
    else: # no frameSTACKING but do SKIP
        env = FrameSkipRAM(env, skip=4)
    return env
	
class ExtractRAMLocations(gym.ObservationWrapper):
    def __init__(self, env, debug_trajectory=False):
        super().__init__(env)
        assert len(env.observation_space.shape) == 1
        self.env = env
        self.debug_trajectory = debug_trajectory
        # necessary for lookup in the dictionary
        self.game_name = self.env.unwrapped.spec.id.split(
            "-")[0].split("No")[0].split("Deterministic")[0].lower()

        assert (self.debug_trajectory and  "pong" in self.game_name) or (not self.debug_trajectory or "pong" not in self.game_name) == 1

        dict_game = atari_dict[self.game_name]
        if "pong" in self.game_name:
            # remove these keys as they're not relevant for this specific game!
            # dict_game.pop("player_x", None)
            # dict_game.pop("enemy_x", None)
            dict_game.pop("enemy_score", None)
            dict_game.pop("player_score", None)
        self.ram_variables_dict = dict_game

        if self.debug_trajectory:
            new_obs_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float64
            )
            self.prev_ball_pos = -100 * np.ones(2)
            self.upper_bound = 209-15
            self.lower_bound = 35

        else:
            new_obs_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(len(self.ram_variables_dict), ),
                dtype=np.float64
            )
        self.observation_space = new_obs_space

        self.observation_space_pong = 4
        self.counter = 0
        self.dump_path = os.path.join(Path.home(), "MA/datadump/ram/trajectory")
        self.offsets = {
            "ball_x": 48,
            "ball_y": 12,
            "enemy_x": 45,
            "enemy_y": 7,
            "player_x": 48,
            "player_y": 5
        }


    def getTrajectoryEndPoint(self, obs_ram):
        ball_x = obs_ram[0]
        ball_y = obs_ram[3]
        # if abs(185 - ball_x) < 5.0 and abs(135 - ball_y) < 3:
        #     pass
        # else:
            # print(f"ball_x {ball_x}, ball_y {ball_y}")

        if np.any(self.prev_ball_pos) < 0: #  or (np.isclose(ball_x, 0) and np.isclose(ball_y, 0)):
            endpoint = ball_y

            ## just for debugging / printing / plotting
            ball_v_x = np.nan
            ball_v_y = np.nan
        else:
            ball_v_x = ball_x - self.prev_ball_pos[0]
            ball_v_y = ball_y - self.prev_ball_pos[1]

            if ball_v_x < -0.5:  # --> ball goes towards enemy and not towards player...
                # ball_v_x *= -1
                # ball_v_y *= -1
                #
                v_quotient_y_x = ball_v_y / ball_v_x
                # endpoint_enemy_y = v_quotient_y_x * (20 - ball_x) + ball_y
                # endpoint = v_quotient_y_x * (140 - 20) + endpoint_enemy_y # ball_y

                endpoint = v_quotient_y_x * (140 - ball_x) + ball_y
            elif ball_v_x == 0.0 and ball_v_y == 0.0:
                endpoint = ball_y
            elif (abs(ball_v_x) > 10) or abs(ball_v_y) > 10:
                endpoint = ball_y
            else:
                if ball_v_x == 0.0:
                    ball_v_x = np.finfo(float).eps
                if ball_v_y == 0.0:
                    ball_v_y = np.finfo(float).eps

                v_quotient_y_x = ball_v_y / ball_v_x

                endpoint = v_quotient_y_x * (140 - ball_x) + ball_y

        endpoint_unprocessed = endpoint
        if endpoint > self.upper_bound:
            endpoint = self.upper_bound - (endpoint - self.upper_bound)
        elif endpoint < self.lower_bound:
            endpoint = self.lower_bound - (endpoint - self.lower_bound)  #  --> positive value
        clipped_val = np.clip(endpoint, 0, 209) # y-value --> range[0, 209]
        if clipped_val != endpoint:
            print(f"unclipped value {endpoint_unprocessed} now: {endpoint}, ball_x {ball_x}, ball_y {ball_y}, prev positions {self.prev_ball_pos[0]}, {self.prev_ball_pos[1]} v: {ball_v_x}, {ball_v_y}")

        self.prev_ball_pos[0] = ball_x
        self.prev_ball_pos[1] = ball_y

        return clipped_val, ball_v_x, ball_v_y

    def observation(self, obs):
        obs_ram = np.array([obs[self.ram_variables_dict["ball_x"]] - self.offsets["ball_x"],
                            obs[self.ram_variables_dict["enemy_y"]] - self.offsets["enemy_y"],
                            obs[self.ram_variables_dict["player_y"]] - self.offsets["player_y"],
                            obs[self.ram_variables_dict["ball_y"]] - self.offsets["ball_y"]], dtype='float64')
        # print(f"{obs_ram[0]} {obs_ram[1]} {obs_ram[2]} {obs_ram[3]}")
        if self.debug_trajectory:
            player_x = obs[self.ram_variables_dict["player_x"]] - self.offsets["player_x"]
            enemy_x = obs[self.ram_variables_dict["enemy_x"]] - self.offsets["enemy_x"]
            trajectory_endpoint, ball_v_x, ball_v_y = self.getTrajectoryEndPoint(obs_ram)
            obs_ram_2 = np.array([obs_ram[2], trajectory_endpoint])  # player_y, endpoint_y

            if self.counter > 10 and self.counter < 310: #  and not torch.cuda.is_available:
                im = self.env.render('rgb_array')
                # center: tuple(COL, ROW) --> here: (x, y)
                # ball
                cv2.circle(im, (int(obs_ram[0]), int(obs_ram[-1])), 6, color=(255, 255, 0), thickness=1)
                # enemy
                cv2.circle(im, (int(enemy_x), int(obs_ram[1])), 2, color=(0, 0, 0), thickness=-1)
                # player
                cv2.circle(im, (int(player_x), int(obs_ram[2])), 2, color=(255, 0, 0), thickness=-1)
                # endpoint
                cv2.circle(im, (int(player_x), int(trajectory_endpoint)), 5, color=(255, 0, 0), thickness=1)
                # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                im = Image.fromarray(im)
                im.save(f"{self.dump_path}_{self.counter}_{obs_ram_2[0]}_{obs_ram_2[1]}_v_{ball_v_x}_{ball_v_y}_posball_{obs_ram[0]}_{obs_ram[-1]}.png")
            self.counter += 1
            obs_ram = obs_ram_2
        # else:
        #     raise SystemError

        if "pong" in self.game_name and not self.debug_trajectory:
            assert len(obs_ram) == self.observation_space_pong
        return obs_ram / 255.0

class FrameSkipRAM(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env) # important otherwise action_space == None

        self._skip = skip
        # self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert len(shp) == 1, "Observation-Space of an environment based on the RAM-State should just be 1 before applying framestacking!"
        # self.observation_space = spaces.Box(
        #     low=env.observation_space.low[0],  # scalar value needed! low respectively high is an array of dim shape...
        #     high=env.observation_space.high[0],
        #     shape=(shp[0], k),
        #     dtype=env.observation_space.dtype
        # )
        self.last_obs = None

    def reset(self, **kwargs):
        # ob = self.env.reset()
        # for _ in range(self.k):
        #     self.frames.append(np.expand_dims(ob, axis=1))
        # return self._get_ob()
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            ob, reward, done, info = self.env.step(action)
            if i == self._skip - 1: # LAST frame of the skipping iteration
                self.last_obs = ob
            total_reward += reward
            if done:
                break
        # self.frames.append(last_obs)
        # self.frames.append(np.expand_dims(ob, axis=1))
        return self.last_obs, total_reward, done, info

    # def _get_ob(self):
    #     assert len(self.frames) == self.k
    #     return np.concatenate(self.frames, axis=1)


class FrameStackRAMFrameSkip(gym.Wrapper):
    def __init__(self, env, k=4, skip=4):
        gym.Wrapper.__init__(self, env) # important otherwise action_space == None

        self.k = k
        self._skip = skip
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert len(shp) == 1, "Observation-Space of an environment based on the RAM-State should just be 1 before applying framestacking!"
        self.observation_space = spaces.Box(
            low=env.observation_space.low[0],  # scalar value needed! low respectively high is an array of dim shape...
            high=env.observation_space.high[0],
            shape=(shp[0], k),
            dtype=env.observation_space.dtype
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(np.expand_dims(ob, axis=1))
        return self._get_ob()

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            ob, reward, done, info = self.env.step(action)
            if i == self._skip -1: # LAST frame of the skipping iteration
                last_obs = ob
            total_reward += reward
            if done:
                break
        # self.frames.append(last_obs)
        self.frames.append(np.expand_dims(ob, axis=1))
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=1)




import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
from PIL import Image
from collections import defaultdict
from atariari.benchmark.ram_annotations import atari_dict
from benchmarking.utils.atari_dict_extended import atari_dict_extended
from benchmarking.utils.process_velocities import threshold_velocities_per_game
from test_atariari.utils.atari_offset_dict import getOffsetDict
from atariari.benchmark.wrapper import InfoWrapper, AtariARIWrapper
from test_atariari.wrapper.atari_wrapper import RemoveScoresFromLabelsPong
from pathlib import Path
from datetime import datetime
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
        """Warp frames to the specified size (height x width)."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.overlay_scores_pong = "pong" in env.spec.id.lower()
        # self.saved_example_obs = False
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.overlay_scores_pong:
            frame[:30, :] = 236 # overlay the upper 30px of Pong
            # if not self.saved_example_obs:
            #     im = Image.fromarray(frame)
            #     now = datetime.now()
            #     im.save(os.path.join(Path.home(), 'MA', 'datadump', 'obs') + "1.png")
            #     self.saved_example_obs = True

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

class FrameStackDeriveVelocitiesOverlapping(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.infos = deque([], maxlen=2)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype)

        self.game_name = env.unwrapped.spec.id.split(
            "-")[0].split("No")[0].split("Deterministic")[0]
        self.additional_labels_dict = atari_dict_extended[self.game_name.lower(
        )]
        self.abs_threshold_invalid = threshold_velocities_per_game[self.game_name]


        self.counter_just_1_info = 0

    def reset(self):
        self.infos.clear()  # --> len=0
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        self.infos.append(info)
        return self._get_ob(), reward, done, self._get_info()

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)

    def _get_info(self):
        label_dict_added = {}
        # if self.prev_frame_info:
        if len(self.infos) == 2:
            # --> derive velocities
            info_latest_frame = self.infos[1]
            label_dict_added, label_dict_added_valid = self.getAdditionalLabelDict(
                latest_labels=info_latest_frame["labels"], #info_latest_frame['labels'],
                second_latest_labels=self.infos[0]["labels"]) # self.prev_frame_info['labels'])
            if label_dict_added_valid:
                labels = {**info_latest_frame['labels'], **label_dict_added}
            else:
                labels = {}
        # combine the additional labels (velocities) and the labels from the very LATEST frame
        elif len(self.infos) == 1:
            info_latest_frame = self.infos[0]
            labels = {**info_latest_frame.get("labels"), **label_dict_added}
            labels = {}
            self.counter_just_1_info += 1
            print(f"counter just 1 info is {self.counter_just_1_info}")
        else:
            sys.exit(1)
        info = {"labels": labels}
        for k, v in info_latest_frame.items():
            if k not in info:
                info.update({k: v})  # e.g. add ale_lives key + val

        return info

    def getVel(self, pos_old, pos_new, frame_diff):
        # if(abs(int(pos_old) - int(pos_new)) > 30):
        #     print(f"new {pos_new}; old {pos_old}")
        valid_flag = True
        vel = int((int(pos_new) - int(pos_old)) / frame_diff)
        if abs(vel) > self.abs_threshold_invalid:
            # INVALID (spawning!)
            valid_flag = False
        return vel, valid_flag

    def getAdditionalLabelDict(self, latest_labels, second_latest_labels):
        labels_added = defaultdict(int)
        labels_added_valid = True  # default
        for k, val in self.additional_labels_dict.items():
            if type(val) is str:
                velocity_value, valid_value = self.getVel(
                    pos_old=second_latest_labels[val], pos_new=latest_labels[val], frame_diff=1)
                if not valid_value:
                    labels_added_valid = False
                labels_added[k] = velocity_value
        return labels_added, labels_added_valid

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

def wrap_deepmind_benchmark(env, dim=84):
    """Configure environment for DeepMind-style Atari.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
    """
    # env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if "NoFrameskip" in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, dim) # includes overlaying the scores of poing
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    env = AtariARIWrapper(env)  # adds the "labels" to the info-dict
    if "pong" in env.unwrapped.spec.id.lower():
        env = RemoveScoresFromLabelsPong(env)
    env = FrameStackDeriveVelocitiesOverlapping(env, 4)
    # DO NOT DO for benchmarking!
    env = ScaledFloatFrame(env)  # TODO: use for dqn?
    return env

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
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    if framestack:
        env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)  # TODO: use for dqn?
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
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    if framestack:
        env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)  # TODO: use for dqn?
    return env
    
def wrap_ram(env, framestack=True, extract_ram=True, debug_trajectory=False, encode_as_bits=False, breakout_keep_blocks=False, input_just_diff=False):
    # ORDER is important
    # FIRST extract rams, then (maybe) stack the observations
    # env = gym_wrappers.Monitor(
    #     env=env,
    #     directory=video_dir,
    #     video_callable=lambda x: True,
    #     force=True)
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    # if "NoFrameskip" in env.spec.id:
        # env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # env = WarpFrame(env, dim)
    # scaling is done by extractRAM
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval

    if extract_ram:
        env = ExtractRAMLocations(env, breakout_keep_blocks=breakout_keep_blocks)
    else:
        env = ScaledFloatFrame(env)  # TODO: use for dqn?

    if framestack:
        env = FrameStackRAMFrameSkip(env, k=4, skip=4, debug_trajectory=debug_trajectory, input_just_diff=input_just_diff)
        framestack = False # set to false later as otherwise problems in bitencoding
    else: # no frameSTACKING but do SKIP
        env = FrameSkipRAM(env, skip=2)
    if encode_as_bits:
        env = BitEncodingWrapper(env, framestack)

    return env

class BitEncodingWrapper(gym.ObservationWrapper):
    def __init__(self, env, framestack=True):
        super().__init__(env)
        # shape[1] != 1 if framestacking (usually ==4)
        if framestack:
            new_observation_space_shape = (env.observation_space.shape[0] * 8 * env.observation_space.shape[1], )
        else:
            new_observation_space_shape = (env.observation_space.shape[0] * 8, )
        # resulting observation layout:
        # s0[0] s0[1] s0[2] s0[3] s1[0] s1[1] s1[2] s1[3] s2[0] s2[1] ...
        # s0: first state variable
        # indexing: first frame, second frame ,... (example for framestacking=True with 4)
        new_observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=new_observation_space_shape,
            dtype=np.bool  #  True / False
        )
        self.observation_space = new_observation_space

    def observation(self, obs):
        if np.max(obs) < 1.0001:
            obs *= 255
        obs = obs.astype(np.uint8)
        return np.unpackbits(obs)


class ExtractRAMLocations(gym.ObservationWrapper):
    def __init__(self, env, breakout_keep_blocks=False):
        super().__init__(env)
        assert len(env.observation_space.shape) == 1
        self.env = env
        # necessary for lookup in the dictionary
        self.game_name = self.env.unwrapped.spec.id.split(
            "-")[0].split("No")[0].split("Deterministic")[0].lower()
        self.breakout_keep_blocks = breakout_keep_blocks

        self.offsets = getOffsetDict(self.game_name)
        dict_game = atari_dict[self.game_name]
        if "pong" in self.game_name:
            # remove these keys as they're not relevant for this specific game!
            dict_game.pop("player_x", None)
            dict_game.pop("enemy_x", None)
            dict_game.pop("enemy_score", None)
            dict_game.pop("player_score", None)
            # self.offsets = {
            #     "ball_x": 48,
            #     "ball_y": 12,
            #     "enemy_x": 45,
            #     "enemy_y": 7,
            #     "player_x": 48,
            #     "player_y": 5
            # }
            self.dump_path = os.path.join(Path.home(), "MA/datadump/ram/pong_traj/")
        if "breakout" in self.game_name:
            dict_game.pop("score", None)
            dict_game.pop("blocks_hit_count", None)
            if not self.breakout_keep_blocks: # --> remove them all (30 variables!)
                for i in range(30):
                    dict_game.pop(f"block_bit_map_{i}", None)
            # self.offsets = dict(ball_x=48,
            #                              ball_y=-11,
            #                              player_x=40)
            self.dump_path = os.path.join(Path.home(), "MA/datadump/ram/breakout_traj/")

        self.ram_variables_dict = dict_game

        new_obs_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.ram_variables_dict),),
            dtype=np.float32
        )
        self.observation_space = new_obs_space

        self.observation_space_pong = 6
        self.counter = 0

    def observation(self, obs):
        if "pong" in self.game_name:
                obs_ram = np.array([np.clip(obs[self.ram_variables_dict["ball_x"]] - self.offsets["ball_x"], 0, 159),
                                    np.clip(obs[self.ram_variables_dict["enemy_y"]] - self.offsets["enemy_y"], 0, 209),
                                    np.clip(obs[self.ram_variables_dict["player_y"]] - self.offsets["player_y"], 0, 209),
                                    np.clip(obs[self.ram_variables_dict["ball_y"]] - self.offsets["ball_y"], 0, 209)], dtype='float64')
                                    # obs[self.ram_variables_dict["enemy_score"]], obs[self.ram_variables_dict["player_score"]]], dtype='float64')
        elif "breakout" in self.game_name:
            obs_ram = np.array([np.clip(obs[self.ram_variables_dict["player_x"]] - self.offsets["player_x"], 0, 159),
                                np.clip(obs[self.ram_variables_dict["ball_x"]] - self.offsets["ball_x"], 0, 159),
                                np.clip(obs[self.ram_variables_dict["ball_y"]] - self.offsets["ball_y"], 0, 209)], dtype='float64')
            if self.breakout_keep_blocks:
                block_bit_list = [obs[self.ram_variables_dict[f"block_bit_map_{i}"]]   for i in range(30)]
                obs_ram = np.append(obs_ram, np.asarray(block_bit_list))

        # print(f"score e {obs_ram[4]} score p {obs_ram[5]}")
        # print(f"{obs_ram[0]} {obs_ram[1]} {obs_ram[2]} {obs_ram[3]}")
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

class FrameStackRAMFrameSkip(gym.Wrapper):
    def __init__(self, env, k=4, skip=4, debug_trajectory=False, input_just_diff=False):
        gym.Wrapper.__init__(self, env) # important otherwise action_space == None

        self.k = k
        self._skip = skip
        self.debug_trajectory_pong = debug_trajectory and "pong" in env.unwrapped.spec.id.lower()
        self.debug_trajectory_breakout = debug_trajectory and "breakout" in env.unwrapped.spec.id.lower()
        self.input_just_diff = input_just_diff and self.debug_trajectory_breakout

        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert len(shp) == 1, "Observation-Space of an environment based on the RAM-State should just be 1 before applying framestacking!"
        if self.debug_trajectory_pong:
            self.observation_space = spaces.Box(
                low=env.observation_space.low[0],  # scalar value needed! low respectively high is an array of dim shape...
                high=env.observation_space.high[0],
                shape=(2,),
                dtype=np.float32
            )
            self.upper_bound = (209-15)/255.
            self.lower_bound = 35 / 255.
            self.obs_traj = np.zeros(self.observation_space.shape, dtype=np.float32)
            self.dump_path = os.path.join(Path.home(), "MA/datadump/ram/pong_traj/")
        elif self.debug_trajectory_breakout:
            self.observation_space = spaces.Box(
                low=env.observation_space.low[0],  # scalar value needed! low respectively high is an array of dim shape...
                high=env.observation_space.high[0],
                shape=(2,),
                dtype=np.float32
            )
            self.upper_bound = (209-93)/255.
            self.side_bound = 8 / 255.
            self.obs_traj = np.zeros(self.observation_space.shape, dtype=np.float32)
            self.dump_path = os.path.join(Path.home(), "MA/datadump/ram/breakout_traj/")
        else:
            self.observation_space = spaces.Box(
                low=env.observation_space.low[0],  # scalar value needed! low respectively high is an array of dim shape...
                high=env.observation_space.high[0],
                shape=(shp[0], k),
                dtype=np.float32
            )
        self.counter = 0
        self.last_endpoint = None

    def _getTrajectoryEndPointPong(self, obs_ram):
        ball_x = obs_ram[-1][0]
        ball_y = obs_ram[-1][3]

        ball_v_x = ball_x - obs_ram[-2][0]
        ball_v_y = ball_y - obs_ram[-2][3]

        if (np.isclose(ball_v_x, 0)) or np.isclose(ball_v_y, 0) or (abs(ball_v_x) > 20/255.) or abs(ball_v_y) > 20/255.:
            if self.last_endpoint:
                endpoint = self.last_endpoint
            else:
                endpoint = ball_y
        else:
            v_quotient_y_x = ball_v_y / ball_v_x

            ## update
            endpoint = v_quotient_y_x * ((140/255.) - ball_x) + ball_y
            self.last_endpoint = endpoint

        return endpoint, ball_v_x, ball_v_y

    def _getTrajectoryEndPointBreakout(self, obs_ram):
        ball_x = obs_ram[-1][1]
        ball_y = obs_ram[-1][2]

        ball_v_x = ball_x - obs_ram[-2][1]
        ball_v_y = ball_y - obs_ram[-2][2]

        if (np.isclose(ball_v_x, 0)) or np.isclose(ball_v_y, 0) or (abs(ball_v_x) > 20/255.) or abs(ball_v_y) > 20/255.:
            if self.last_endpoint:
                endpoint = self.last_endpoint
            else:
                endpoint = ball_x
        else:
            v_quotient_x_y = ball_v_x / ball_v_y

            endpoint = v_quotient_x_y * ((190/255.) - ball_y) + ball_x
            self.last_endpoint = endpoint

        return endpoint, ball_v_x, ball_v_y

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(np.expand_dims(ob, axis=1))
        return self._get_ob(reset=True)

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
        self.frames.append(np.expand_dims(ob, axis=1))
        return self._get_ob(), total_reward, done, info

    def _getBoundedValuePong(self, value):
        value_unprocessed = value
        if value > self.upper_bound:
            value = self.upper_bound - (value - self.upper_bound)
        elif value < self.lower_bound:
            value = self.lower_bound - (value - self.lower_bound)  #  --> positive value
        clipped_val = np.clip(value, 0, 209/255.) # y-value --> range[0, 209]
        return clipped_val

    def _getBoundedValueBreakout(self, value):
        value_unprocessed = value
        if value < self.side_bound:
            value = self.side_bound - (value - self.side_bound)
        if value > (159-self.side_bound):
            value = 2*(159-self.side_bound) - value
        clipped_val = np.clip(value, 0, 159/255.) # x-value --> range[0, 159]
        return clipped_val

    def _get_ob(self, reset=False):
        assert len(self.frames) == self.k
        if self.debug_trajectory_pong:
            player_y = self.frames[-1][2]
            if reset:
                # weird workaround, if np-array is created here --> often problems with shape (2,1) instead of (2,)...
                self.obs_traj[0] = player_y
                self.obs_traj[1] = player_y
            else:
                endpoint, ball_v_x, ball_v_y = self._getTrajectoryEndPointPong(self.frames)
                self.obs_traj[0] = self._getBoundedValuePong(endpoint)
                self.obs_traj[1] = player_y # self._getBoundedValue(player_y)
                # self.obs_traj[2] = self.frames[-1][4]
                # self.obs_traj[3] = self.frames[-1][5]
                # do not save plots on GPU (doesn't make sense with multiprocessing of envs/workers)
                if not torch.cuda.is_available() and self.counter < 400: #  and not torch.cuda.is_available:
                    obs_ram = self.frames[-1]
                    enemy_x = 20
                    player_x = 160-20
                    im = self.env.render('rgb_array')
                    # center: tuple(COL, ROW) --> here: (x, y)
                    # ball
                    cv2.circle(im, (int(obs_ram[0]*255), int(255*obs_ram[3])), 6, color=(255, 255, 0), thickness=1)
                    # enemy
                    cv2.circle(im, (int(enemy_x), int(255*obs_ram[1])), 2, color=(0, 0, 0), thickness=-1)
                    # player
                    cv2.circle(im, (int(player_x), int(255*obs_ram[2])), 2, color=(255, 0, 0), thickness=-1)
                    # endpoint
                    cv2.circle(im, (int(player_x), int(255*self.obs_traj[0])), 5, color=(255, 0, 0), thickness=3)
                    # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                    im = Image.fromarray(im)
                    im.save(f"{self.dump_path}{self.counter}_{int(255*self.obs_traj[0])}_{int(255*self.obs_traj[1])}_v_{ball_v_x*255}_{ball_v_y*255}_posball_{255*obs_ram[0]}_{255*obs_ram[3]}.png")
                self.counter += 1
            assert self.obs_traj.shape==(2,)

            return self.obs_traj
        if self.debug_trajectory_breakout:
            player_x = self.frames[-1][0]
            if reset:
                # weird workaround, if np-array is created here --> often problems with shape (2,1) instead of (2,)...
                if self.input_just_diff:
                    self.obs_traj[0] = player_x
                else:
                    self.obs_traj[0] = player_x
                    self.obs_traj[1] = player_x
            else:
                endpoint, ball_v_x, ball_v_y = self._getTrajectoryEndPointBreakout(self.frames)
                if self.input_just_diff:
                    diff = self.getDifference(endpoint=endpoint, player_position=player_x)
                    self.obs_traj[0] = abs(diff)
                    if diff > 0:
                        self.obs_traj[1] = 1
                    else:
                        self.obs_traj[1] = 0
                else:
                    self.obs_traj[0] = self._getBoundedValueBreakout(endpoint)
                    self.obs_traj[1] = player_x # self._getBoundedValue(player_y)
                # do not save plots on GPU (doesn't make sense with multiprocessing of envs/workers)
                if not torch.cuda.is_available()  and self.counter < 500: #  and not torch.cuda.is_available:
                    obs_ram = self.frames[-1]
                    player_y = 209-20
                    im = self.env.render('rgb_array')
                    # center: tuple(COL, ROW) --> here: (x, y)
                    # ball
                    cv2.circle(im, (int(obs_ram[1]*255), int(255*obs_ram[2])), 6, color=(255, 255, 0), thickness=1)
                    # endpoint
                    if self.input_just_diff:
                        # TODO: create line from endpoint to player
                        # cv2.circle(im, (int(255*self.obs_traj[0]), player_y), 5, color=(255, 0, 0), thickness=3)
                        if self.obs_traj[1] > 0.99:
                            val_add = self.obs_traj[0]
                        elif np.isclose(self.obs_traj[1], 0):
                            if self.obs_traj[0] > 0:
                                val_add = -self.obs_traj[0]
                            else:
                                val_add = 0
                        second_point = (255*(self.frames[-1][0] + val_add), player_y)
                        cv2.line(im, (int(255*self.frames[-1][0]), player_y), second_point, color=(0, 255, 0), thickness=2)
                    else:
                        cv2.circle(im, (int(255*self.obs_traj[0]), player_y), 5, color=(255, 0, 0), thickness=3)
                        print(f"endoint: ({255*self.obs_traj[0]}, {player_y}); unbounded: {255*endpoint}")
                        # player
                        cv2.circle(im, (int(255*obs_ram[0]), player_y), 3, color=(255, 255, 0), thickness=-1)
                    # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                    im = Image.fromarray(im)
                    path = f"{self.dump_path}{self.counter}_{int(255*self.obs_traj[0])}_{int(255*self.obs_traj[1])}_v_{ball_v_x*255}_{ball_v_y*255}_posball_{255*obs_ram[1]}_{255*obs_ram[2]}.png"
                    im.save(path)
                self.counter += 1
            # assert self.obs_traj.shape==(2,)

            return self.obs_traj

        else:
            return np.concatenate(self.frames, axis=1)

    def getDifference(self, endpoint, player_position):
        # TODO: how to scale??
        return endpoint - player_position



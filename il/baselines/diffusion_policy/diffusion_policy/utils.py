import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from h5py import Dataset, File, Group
from torch.utils.data.sampler import Sampler


class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.
    Resampling from it until a specified number of iterations have been sampled
    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


TARGET_KEY_TO_SOURCE_KEY = {
    "states": "env_states",
    "observations": "obs",
    "success": "success",
    "next_observations": "obs",
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    "actions": "actions",
}


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def load_traj_hdf5(path, num_traj=None):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
        keys = keys[:num_traj]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    print("Loaded")
    return ret


def load_demo_dataset(
    path, keys=["observations", "actions"], num_traj=None, concat=True
):
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = raw_data["traj_0"]
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in raw_data]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[:-1] for t in dataset[target_key]], axis=0
                )
            elif target_key in ["next_observations", "next_states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[1:] for t in dataset[target_key]], axis=0
                )
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print("Load", target_key, dataset[target_key].shape)
        else:
            print(
                "Load",
                target_key,
                len(dataset[target_key]),
                type(dataset[target_key][0]),
            )
    return dataset


def convert_obs(obs, concat_fn, transpose_fn, state_obs_extractor, depth = True):
    img_dict = obs["sensor_data"]
    ls = ["rgb"]
    if depth:
        ls = ["rgb", "depth"]

    new_img_dict = {
        key: transpose_fn(
            concat_fn([v[key] for v in img_dict.values()])
        )  # (C, H, W) or (B, C, H, W)
        for key in ls
    }
    if "depth" in new_img_dict and isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
        new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

    # Unified version
    states_to_stack = state_obs_extractor(obs)
    for j in range(len(states_to_stack)):
        if states_to_stack[j].dtype == np.float64:
            states_to_stack[j] = states_to_stack[j].astype(np.float32)
    try:
        state = np.hstack(states_to_stack)
    except:  # dirty fix for concat trajectory of states
        state = np.column_stack(states_to_stack)
    if state.dtype == np.float64:
        for x in states_to_stack:
            print(x.shape, x.dtype)
        import pdb

        pdb.set_trace()

    out_dict = {
        "state": state,
        "rgb": new_img_dict["rgb"],
    }

    if "depth" in new_img_dict:
        out_dict["depth"] = new_img_dict["depth"]


    return out_dict


def build_obs_space(env, depth_dtype, state_obs_extractor):
    # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env
    obs_space = env.observation_space

    # Unified version
    state_dim = sum([v.shape[0] for v in state_obs_extractor(obs_space)])

    single_img_space = next(iter(env.observation_space["image"].values()))
    h, w, _ = single_img_space["rgb"].shape
    n_images = len(env.observation_space["image"])

    return spaces.Dict(
        {
            "state": spaces.Box(
                -float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32
            ),
            "rgb": spaces.Box(0, 255, shape=(n_images * 3, h, w), dtype=np.uint8),
            "depth": spaces.Box(
                -float("inf"), float("inf"), shape=(n_images, h, w), dtype=depth_dtype
            ),
        }
    )


def build_state_obs_extractor(env_id):
    # NOTE: You can tune/modify state observations specific to each environment here as you wish. By default we include all data
    # but in some use cases you might want to exclude e.g. obs["agent"]["qvel"] as qvel is not always something you query in the real world.
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())


def obs_at_timestep(obs, t=0):
    """Slice a demo trajectory observation dict to a single timestep."""
    if isinstance(obs, dict):
        return {k: obs_at_timestep(v, t) for k, v in obs.items()}
    return np.asarray(obs[t])


def nested_obs_space_from_sample(sample):
    """Build a gym Dict/Box tree matching a nested demo observation sample."""
    if isinstance(sample, dict):
        return spaces.Dict(
            {k: nested_obs_space_from_sample(v) for k, v in sample.items()}
        )
    sample = np.asarray(sample)
    if sample.dtype == np.bool_:
        sample = sample.astype(np.float32)
    if sample.dtype == np.uint8:
        lo, hi = 0, 255
    else:
        lo, hi = -float("inf"), float("inf")
    return spaces.Box(lo, hi, shape=sample.shape, dtype=sample.dtype)


def demo_nested_obs_space(h5_path, num_traj=1):
    """Observation-space tree for reorder_keys, inferred from demo HDF5 (no sim env)."""
    raw = load_traj_hdf5(h5_path, num_traj=num_traj)
    traj_key = sorted(raw.keys(), key=lambda x: int(x.split("_")[-1]))[0]
    return nested_obs_space_from_sample(obs_at_timestep(raw[traj_key]["obs"], 0))


def demo_visual_flags(h5_path, obs_mode):
    """Whether rgb/depth tensors are present in the demo and requested by obs_mode."""
    include_rgb = "rgb" in obs_mode
    include_depth = "depth" in obs_mode
    if h5_path.endswith(".h5"):
        raw = load_traj_hdf5(h5_path, num_traj=1)
        traj_key = next(iter(raw))
        cam = next(iter(raw[traj_key]["obs"]["sensor_data"].values()))
        include_rgb = include_rgb or "rgb" in cam
        include_depth = include_depth or "depth" in cam
    return include_rgb, include_depth


def mock_rgbd_agent_env(dataset, obs_horizon, include_rgb, include_depth):
    """Minimal env stand-in for Agent init when no ManiSkill env is available."""
    proc = dataset.trajectories["observations"][0]
    act_dim = dataset.trajectories["actions"][0].shape[-1]
    state_dim = proc["state"].shape[-1]
    obs_spaces = {
        "state": spaces.Box(
            -float("inf"), float("inf"), shape=(obs_horizon, state_dim), dtype=np.float32
        ),
    }
    if include_rgb:
        c, h, w = proc["rgb"].shape[1:]
        # Agent reads channel count from the last dim (matches FrameStack + HWC layout).
        obs_spaces["rgb"] = spaces.Box(
            0, 255, shape=(obs_horizon, h, w, c), dtype=np.uint8
        )
    if include_depth:
        depth = proc["depth"]
        if depth.ndim == 4:
            c, h, w = depth.shape[1:]
        else:
            h, w = depth.shape[-2:]
            c = 1
        obs_spaces["depth"] = spaces.Box(
            -float("inf"), float("inf"), shape=(obs_horizon, h, w, c), dtype=np.float16
        )

    class _MockEnv:
        single_observation_space = spaces.Dict(obs_spaces)
        single_action_space = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)

    return _MockEnv()

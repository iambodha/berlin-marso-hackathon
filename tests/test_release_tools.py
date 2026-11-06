import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

from scripts.report_results import summarize

spec = importlib.util.spec_from_file_location('rollout', 'render_rollout.py')
rollout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollout)


class Flag:
    def __init__(self, value): self.value = value
    def any(self): return self.value


class Env:
    def __init__(self, terminate, truncate=False):
        self.steps = 0; self.terminate = terminate; self.truncate = truncate
    def reset(self, seed): return 0, {}
    def step(self, action):
        self.steps += 1
        return self.steps, 0, Flag(self.terminate), Flag(self.truncate), {}


class Agent:
    open_loop = False
    def reset(self): self.reset_called = True
    def act(self, obs, deterministic): return 0


class Chunk:
    shape = (1, 4)
    def __getitem__(self, key): return 0


class ChunkAgent(Agent):
    open_loop = True
    def get_action(self, obs): return Chunk()


class ReleaseTests(unittest.TestCase):
    def test_saved_medium_result(self):
        path = 'il/baselines/diffusion_policy/runs/warehouse_combined_obs8_30k/results.json'
        report = summarize(path)
        self.assertEqual(report['best_iteration'], 20000)
        self.assertEqual(report['best_sort_accuracy'], .203125)
        self.assertEqual(report['last_sort_accuracy'], .1875)

    def test_termination_and_step_limit_for_both_policy_interfaces(self):
        for cls in (Agent, ChunkAgent):
            for terminated, truncated, expected in ((True, False, 1), (False, True, 1), (False, False, 4)):
                env, agent = Env(terminated, truncated), cls()
                frames = rollout._rollout_frames(env, agent, 'cpu', 5, 2, 42,
                    convert=lambda obs, device: obs, parse_frame=lambda obs, parcels: obs)
                self.assertEqual(env.steps, expected)
                self.assertEqual(len(frames), expected+1)
                self.assertTrue(agent.reset_called)

    def test_help_and_missing_checkpoint_without_gpu_dependencies(self):
        help_result = subprocess.run([sys.executable, 'render_rollout.py', '--help'], capture_output=True)
        self.assertEqual(help_result.returncode, 0)
        result = subprocess.run([sys.executable, 'render_rollout.py', '--checkpoint', 'does-not-exist.pt'], capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn(b'Checkpoint not found', result.stderr)

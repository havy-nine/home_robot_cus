# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import time
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from utils.env_utils import create_ovmm_env_fn
from utils.metrics_utils import get_stats_from_episode_metrics

if TYPE_CHECKING:
    from habitat.core.dataset import BaseEpisode
    from habitat.core.vector_env import VectorEnv

    from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
    from home_robot.core.abstract_agent import Agent


class EvaluationType(Enum):
    """Whether we run local or remote evaluation."""

    LOCAL = "local"
    LOCAL_VECTORIZED = "local_vectorized"
    REMOTE = "remote"


class OVMMEvaluator(PPOTrainer):
    def __init__(self, eval_config: DictConfig, data_dir=None) -> None:
        self.metrics_save_freq = eval_config.EVAL_VECTORIZED.metrics_save_freq
        self.results_dir = os.path.join(
            eval_config.DUMP_LOCATION, "results", eval_config.EXP_NAME
        )
        self.videos_dir = os.path.join(eval_config.DUMP_LOCATION, "video")
        self.data_dir = data_dir
        if self.data_dir:
            os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        self.current_scene = None  # 장면 전환 감지를 위한 변수

        super().__init__(eval_config)

    def save_first_person_image(
        self, scene_id: str, episode_id: str, image: np.ndarray, step: int = 0
    ) -> None:
        """Save first_person_image to scene-specific folder."""
        if self.data_dir is None:
            return
        scene_folder = os.path.join(
            self.data_dir, f"scene_{scene_id.split('/')[-1].split('.')[0]}"
        )
        os.makedirs(scene_folder, exist_ok=True)
        image_path = os.path.join(scene_folder, f"episode_{episode_id}_step_{step}.png")
        Image.fromarray(image).save(image_path)
        print(f"Saved first_person_image to {image_path}")

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        env_num_episodes = self._env.number_of_episodes
        if num_episodes is None:
            num_episodes = env_num_episodes
        else:
            assert num_episodes <= env_num_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, env_num_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}
        episode_key_order: list = []
        count_episodes: int = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations, done = self._env.reset(), False
            current_episode = self._env.get_current_episode()
            agent.reset()
            self._check_set_planner_vis_dir(agent, current_episode)

            # 장면 전환 감지
            scene_id = current_episode.scene_id
            if scene_id != self.current_scene:
                print(f"Scene changed to {scene_id.split('/')[-1].split('.')[0]}")
                self.current_scene = scene_id

            current_episode_key = (
                f"{scene_id.split('/')[-1].split('.')[0]}_"
                f"{current_episode.episode_id}"
            )
            episode_key_order.append(current_episode_key)
            current_episode_metrics = {}

            # 초기 first_person_image 저장
            if "rgb" in observations:
                self.save_first_person_image(
                    scene_id, current_episode.episode_id, observations["rgb"], step=0
                )

            step = 1  # 스텝 카운터
            while not done:
                action, info, _ = agent.act(observations)
                observations, done, hab_info = self._env.apply_action(action, info)
                # 매 스텝마다 first_person_image 저장
                if "rgb" in observations:
                    self.save_first_person_image(
                        scene_id,
                        current_episode.episode_id,
                        observations["rgb"],
                        step=step,
                    )
                step += 1

                if "skill_done" in info and info["skill_done"] != "":
                    metrics = extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    current_episode_metrics = {
                        **metrics_at_skill_end,
                        **current_episode_metrics,
                    }
                    if "goal_name" in info:
                        current_episode_metrics["goal_name"] = info["goal_name"]

            metrics = extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            if "goal_name" in info:
                current_episode_metrics["goal_name"] = info["goal_name"]

            episode_metrics[current_episode_key] = current_episode_metrics
            if len(episode_metrics) % self.metrics_save_freq == 0:
                aggregated_metrics = self._aggregate_metrics(episode_metrics)
                self._write_results(episode_metrics, aggregated_metrics)

            count_episodes += 1
            pbar.update(1)

        self._env.close()

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)
        print(episode_key_order)
        return average_metrics

    def _aggregate_metrics(self, episode_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Aggregates metrics tracked by environment."""
        aggregated_metrics = defaultdict(list)
        metrics = set(
            [
                k
                for metrics_per_episode in episode_metrics.values()
                for k in metrics_per_episode
                if k != "goal_name"
            ]
        )
        for v in episode_metrics.values():
            for k in metrics:
                if k in v:
                    aggregated_metrics[f"{k}/total"].append(v[k])

        aggregated_metrics = dict(
            sorted(
                {
                    k2: v2
                    for k1, v1 in aggregated_metrics.items()
                    for k2, v2 in {
                        f"{k1}/mean": np.mean(v1),
                        f"{k1}/min": np.min(v1),
                        f"{k1}/max": np.max(v1),
                    }.items()
                }.items()
            )
        )

        return aggregated_metrics

    def _write_results(
        self, episode_metrics: Dict[str, Dict], aggregated_metrics: Dict[str, float]
    ) -> None:
        """Writes metrics tracked by environment to a file."""
        with open(f"{self.results_dir}/aggregated_results.json", "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        with open(f"{self.results_dir}/episode_results.json", "w") as f:
            json.dump(episode_metrics, f, indent=4)

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates the agent in the local environment.

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        env_num_episodes = self._env.number_of_episodes
        if num_episodes is None:
            num_episodes = env_num_episodes
        else:
            assert num_episodes <= env_num_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, env_num_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}
        episode_key_order: list = []
        print(f"num_episodes : {num_episodes}")
        print(f"_env.number_of_episodes: {env_num_episodes}")
        count_episodes: int = 0

        pbar = tqdm(total=num_episodes)
        # print("In Evaluator, num_episodes: ", num_episodes)
        while count_episodes < num_episodes:
            observations, done = self._env.reset(), False
            current_episode = self._env.get_current_episode()
            agent.reset()
            self._check_set_planner_vis_dir(agent, current_episode)

            current_episode_key = (
                f"{current_episode.scene_id.split('/')[-1].split('.')[0]}_"
                f"{current_episode.episode_id}"
            )
            print(f"in projects/habitat_ovmm/evaluator.py, {current_episode_key}")
            episode_key_order.append(current_episode_key)
            current_episode_metrics = {}
            obs_data = [observations]
            while not done:
                action, info, _ = agent.act(observations)
                observations, done, hab_info = self._env.apply_action(action, info)
                # print("In evaluator, _env.apply_action : ", self._env.apply_action(action, info)[1])
                if self.data_dir:
                    obs_data.append(observations)
                if "skill_done" in info and info["skill_done"] != "":
                    metrics = extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    current_episode_metrics = {
                        **metrics_at_skill_end,
                        **current_episode_metrics,
                    }
                    if "goal_name" in info:
                        current_episode_metrics["goal_name"] = info["goal_name"]

            if self.data_dir:
                import pickle

                data_episode_path = os.path.join(self.data_dir, current_episode_key)
                os.makedirs(data_episode_path, exist_ok=True)
                with open(os.path.join(data_episode_path, "obs_data.pkl"), "wb") as f:
                    pickle.dump(obs_data, f)

            metrics = extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            if "goal_name" in info:
                current_episode_metrics["goal_name"] = info["goal_name"]

            episode_metrics[current_episode_key] = current_episode_metrics
            if len(episode_metrics) % self.metrics_save_freq == 0:
                aggregated_metrics = self._aggregate_metrics(episode_metrics)
                self._write_results(episode_metrics, aggregated_metrics)

            count_episodes += 1
            pbar.update(1)

        self._env.close()

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)
        print(episode_key_order)
        return average_metrics

    def remote_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates the agent in the remote environment.

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        # The modules imported below are specific to challenge remote evaluation.
        # These modules are not part of the home-robot repository.
        import pickle
        import time

        import grpc

        try:
            import evaluation_pb2
            import evaluation_pb2_grpc
        except ImportError:
            from home_robot_hw.utils.eval_ai import evaluation_pb2, evaluation_pb2_grpc

        # Wait for the remote environment to be up and running
        time.sleep(60)

        def grpc_dumps(entity):
            return pickle.dumps(entity)

        def grpc_loads(entity):
            return pickle.loads(entity)

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(
            target=env_address_port,
            compression=grpc.Compression.Gzip,
            options=[
                (
                    "grpc.max_receive_message_length",
                    -1,
                )  # Unlimited message length that the channel can receive
            ],
        )
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        stub.init_env(evaluation_pb2.Package())

        env_num_episodes = grpc_loads(
            stub.number_of_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        if num_episodes is None:
            num_episodes = env_num_episodes
        else:
            assert num_episodes <= env_num_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, env_num_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}

        count_episodes: int = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations, done = (
                grpc_loads(stub.reset(evaluation_pb2.Package()).SerializedEntity),
                False,
            )
            current_episode = grpc_loads(
                stub.get_current_episode(evaluation_pb2.Package()).SerializedEntity
            )
            agent.reset()
            self._check_set_planner_vis_dir(agent, current_episode)

            current_episode_key = (
                f"{current_episode.scene_id.split('/')[-1].split('.')[0]}_"
                f"{current_episode.episode_id}"
            )
            current_episode_metrics = {}

            while not done:
                action, info, _ = agent.act(observations)
                observations, done, hab_info = grpc_loads(
                    stub.apply_action(
                        evaluation_pb2.Package(
                            SerializedEntity=grpc_dumps((action, info))
                        )
                    ).SerializedEntity
                )

                # record metrics if the current skill finishes
                if hab_info is not None:
                    if "skill_done" in info and info["skill_done"] != "":
                        metrics = extract_scalars_from_info(hab_info)
                        metrics_at_skill_end = {
                            f"{info['skill_done']}." + k: v for k, v in metrics.items()
                        }
                        current_episode_metrics = {
                            **metrics_at_skill_end,
                            **current_episode_metrics,
                        }
                        if "goal_name" in info:
                            current_episode_metrics["goal_name"] = info["goal_name"]

            metrics = extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            if "goal_name" in info:
                current_episode_metrics["goal_name"] = info["goal_name"]

            episode_metrics[current_episode_key] = current_episode_metrics
            if len(episode_metrics) % self.metrics_save_freq == 0:
                aggregated_metrics = self._aggregate_metrics(episode_metrics)
                self._write_results(episode_metrics, aggregated_metrics)

            count_episodes += 1
            pbar.update(1)

        stub.close(evaluation_pb2.Package())
        stub.evalai_update_submission(evaluation_pb2.Package())

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def evaluate(
        self,
        agent: "Agent",
        num_episodes: Optional[int] = 100,
        evaluation_type: str = "local",
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """
        print(
            f"projects/habitat_ovmm/evaluator.py, evaluation_type : {evaluation_type}"
        )
        if evaluation_type == EvaluationType.LOCAL.value:
            print("1")
            self._env = create_ovmm_env_fn(self.config)
            print(f"print(self._env) : {self._env}")
            return self.local_evaluate(agent, num_episodes)
        elif evaluation_type == EvaluationType.LOCAL_VECTORIZED.value:
            print("2")
            self._env = create_ovmm_env_fn(self.config)
            return self.local_evaluate_vectorized(agent, num_episodes)
        elif evaluation_type == EvaluationType.REMOTE.value:
            print("3")
            self._env = None
            return self.remote_evaluate(agent, num_episodes)
        else:
            raise ValueError(
                "Invalid evaluation type. Please choose from 'local', 'local_vectorized', 'remote'"
            )

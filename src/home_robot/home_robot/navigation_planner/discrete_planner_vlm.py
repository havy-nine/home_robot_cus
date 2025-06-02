# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import math
import os
import shutil
import socket
import time
from io import BytesIO
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
from natsort import natsorted
from PIL import Image

import home_robot.utils.pose as pu
from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)
from home_robot.utils.geometry import xyt_global_to_base

from .fmm_planner import FMMPlanner

CM_TO_METERS = 0.01


def add_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    h, w = mat.shape
    new_mat = np.zeros((h + 2, w + 2)) + value
    new_mat[1 : h + 1, 1 : w + 1] = mat
    return new_mat


def remove_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    return mat[value:-value, value:-value]


class DiscretePlanner:
    """
    This class translates planner inputs into a discrete low-level action
    using an FMM planner.

    This is a wrapper used to navigate to a particular object/goal location.
    """


class DiscretePlanner:
    def __init__(
        self,
        turn_angle: float,
        collision_threshold: float,
        step_size: int,
        obs_dilation_selem_radius: int,
        goal_dilation_selem_radius: int,
        map_size_cm: int,
        map_resolution: int,
        visualize: bool,
        print_images: bool,
        dump_location: str,
        exp_name: str,
        min_goal_distance_cm: float = 50.0,
        min_obs_dilation_selem_radius: int = 1,
        agent_cell_radius: int = 1,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
        goal_tolerance: float = 0.01,
        discrete_actions: bool = True,
        continuous_angle_tolerance: float = 30.0,
        llava_host: str = "localhost",  # LLaVA 서버 호스트
        llava_port: int = 9999,  # LLaVA 서버 포트 (9999로 변경)
    ):
        # --- 추가 시작 ---
        self.position_history = []  # 위치 이력 (x, y)
        self.stuck_threshold = 5  # 정체 판단 타임스텝 임계값 (예: 5단계)
        self.stuck_counter = 0  # 정체 카운터
        self.backtrack_mode = False  # 이전 위치로 돌아가는 모드
        self.backtrack_target = None  # 돌아갈 이전 위치
        self.llava_direction = None  # LLaVA 서버에서 받은 방향
        self.llava_host = llava_host  # LLaVA 서버 호스트
        self.llava_port = llava_port  # LLaVA 서버 포트
        # --- 추가 끝
        """
        Arguments:
            turn_angle (float): agent turn angle (in degrees)
            collision_threshold (float): forward move distance under which we
             consider there's a collision (in meters)
            obs_dilation_selem_radius: radius (in cells) of obstacle dilation
             structuring element
            obs_dilation_selem_radius: radius (in cells) of goal dilation
             structuring element
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            visualize: if True, render planner internals for debugging
            print_images: if True, save visualization as images
        """
        self.discrete_actions = discrete_actions
        self.visualize = visualize
        self.print_images = print_images
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.map_shape = (
            self.map_size_cm // self.map_resolution,
            self.map_size_cm // self.map_resolution,
        )
        self.turn_angle = turn_angle
        self.collision_threshold = collision_threshold
        self.step_size = step_size
        self.start_obs_dilation_selem_radius = obs_dilation_selem_radius
        self.goal_dilation_selem_radius = goal_dilation_selem_radius
        self.min_obs_dilation_selem_radius = min_obs_dilation_selem_radius
        self.agent_cell_radius = agent_cell_radius
        self.goal_tolerance = goal_tolerance
        self.continuous_angle_tolerance = continuous_angle_tolerance

        self.vis_dir = None
        self.collision_map = None
        self.visited_map = None
        self.col_width = None
        self.last_pose = None
        self.curr_pose = None
        self.last_action = None
        self.timestep = 0
        self.curr_obs_dilation_selem_radius = None
        self.obs_dilation_selem = None
        self.min_goal_distance_cm = min_goal_distance_cm
        self.dd = None

        self.map_downsample_factor = map_downsample_factor
        self.map_update_frequency = map_update_frequency

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape)
        self.visited_map = np.zeros(self.map_shape)
        self.col_width = 1
        self.last_pose = None
        self.curr_pose = [
            self.map_size_cm / 100.0 / 2.0,
            self.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self.timestep = 1
        self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius
        self.obs_dilation_selem = skimage.morphology.disk(
            self.curr_obs_dilation_selem_radius
        )
        self.goal_dilation_selem = skimage.morphology.disk(
            self.goal_dilation_selem_radius
        )
        # --- 추가 시작 ---
        self.position_history = [(self.curr_pose[0], self.curr_pose[1])]
        self.stuck_counter = 0
        self.backtrack_mode = False
        self.backtrack_target = None
        self.llava_direction = None
        # --- 추가 끝

    def set_vis_dir(self, scene_id: str, episode_id: str):
        self.vis_dir = os.path.join(self.default_vis_dir, f"{scene_id}_{episode_id}")
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def disable_print_images(self):
        self.print_images = False

    # def get_latest_image(self) -> np.ndarray:
    #     """vis_dir에서 가장 최근 이미지를 가져옵니다."""
    #     # vis_dir에서 snapshot_*.png 파일 목록을 가져옴
    #     image_files = natsorted(glob.glob(os.path.join(self.vis_dir, "snapshot_*.png")))
    #     if not image_files:
    #         raise FileNotFoundError(f"No snapshot images found in {self.vis_dir}")

    #     # 가장 최근 파일 (natsorted로 정렬했으므로 마지막 파일)
    #     latest_image_path = image_files[-1]
    #     print(f"Using latest image: {latest_image_path}")

    #     # 이미지 로드
    #     image = cv2.imread(latest_image_path)
    #     x1 = 50
    #     y1 = 50
    #     x2 = 390
    #     y2 = 520
    #     cropped_img = image(cv2.img[y1:y2, x1:x2])
    #     cv2.imwrite(f"/home/intern/cheon/home-robot/cropped.png", cropped_img)
    #     if cropped_img is None:
    #         raise ValueError(f"Failed to load image from {latest_image_path}")
    #     return cropped_img
    def get_latest_image(self) -> np.ndarray:
        """vis_dir에서 가장 최근 이미지를 가져옵니다."""
        # vis_dir에서 snapshot_*.png 파일 목록을 가져옴
        image_files = natsorted(glob.glob(os.path.join(self.vis_dir, "snapshot_*.png")))
        if not image_files:
            raise FileNotFoundError(f"No snapshot images found in {self.vis_dir}")

        # 가장 최근 파일 (natsorted로 정렬했으므로 마지막 파일)
        latest_image_path = image_files[-1]
        print(f"Using latest image: {latest_image_path}")

        # 이미지 로드
        image = cv2.imread(latest_image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {latest_image_path}")

        # 크롭 좌표 설정
        x1 = 50
        y1 = 50
        x2 = 390
        y2 = 520

        # 이미지 크기 확인
        height, width = image.shape[:2]
        if x2 > width or y2 > height or x1 < 0 or y1 < 0:
            raise ValueError(
                f"Crop coordinates out of bounds: image size ({width}, {height}), crop ([{x1}:{x2}, {y1}:{y2}])"
            )

        # 이미지 크롭 (NumPy 배열 슬라이싱 사용)
        cropped_img = image[y1:y2, x1:x2]

        # 크롭된 이미지가 비어 있는지 확인
        if cropped_img.size == 0:
            raise ValueError(f"Cropped image is empty: [{x1}:{x2}, {y1}:{y2}]")

        # 크롭된 이미지 저장
        # cv2.imwrite("/home/intern/cheon/home-robot/cropped.png", cropped_img)
        # print('save img successfully.')
        return cropped_img

    def send_image_to_llava(self, image: np.ndarray) -> str:
        """이미지를 LLaVA 서버로 전송하고 응답을 받습니다."""
        try:
            # 이미지를 PNG 형식으로 변환
            _, buffer = cv2.imencode(".png", image)
            image_bytes = buffer.tobytes()

            # 소켓 생성
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.llava_host, self.llava_port))
            print(f"Connected to LLaVA server at {self.llava_host}:{self.llava_port}")

            # 이미지 데이터 전송 (이미지 크기 전송 단계 제거)
            client_socket.sendall(image_bytes)
            print(f"Sent image to LLaVA server (size: {len(image_bytes)} bytes)")

            # 클라이언트 소켓 종료 (서버가 데이터 수신 완료를 인식하도록)
            client_socket.shutdown(socket.SHUT_WR)

            # 서버로부터 응답 수신
            response = b""
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                response += data
            response_str = response.decode("utf-8")
            print(f"Received response from LLaVA server: {response_str}")

            # 소켓 닫기
            client_socket.close()

            # 응답 처리
            if response_str == "error":
                print(
                    "LLaVA server returned an error. Using default direction: backward"
                )
                return "backward"
            return response_str

        except Exception as e:
            print(f"Error communicating with LLaVA server: {e}")
            return "backward"  # 에러 발생 시 기본 방향 반환

    def plan(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        frontier_map: np.ndarray,
        sensor_pose: np.ndarray,
        found_goal: bool,
        debug: bool = True,
        use_dilation_for_stg: bool = False,
        timestep: int = None,
    ) -> Tuple[DiscreteNavigationAction, np.ndarray]:
        if timestep is not None:
            self.timestep = timestep

        self.last_pose = self.curr_pose
        obstacle_map = np.rint(obstacle_map)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        start = [
            int(start_y * 100.0 / self.map_resolution - gx1),
            int(start_x * 100.0 / self.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, obstacle_map.shape)
        start = np.array(start)
        self.curr_pose = [start_x, start_y, start_o]
        self.visited_map[gx1:gx2, gy1:gy2][
            start[0] : start[0] + 1, start[1] : start[1] + 1
        ] = 1

        if debug:
            print()
            print("--- Planning ---")
            print("Found goal:", found_goal)
            print("Goal points provided:", np.any(goal_map > 0))

        current_position = (start_x, start_y)
        self.position_history.append(current_position)

        # 정체 상태 확인: 직전 위치와 현재 위치만 비교
        debug = True
        if len(self.position_history) >= 2:  # 최소 2개의 위치가 있어야 비교 가능
            last_position = self.position_history[-2]
            current_position = self.position_history[-1]
            distance_moved = np.linalg.norm(
                np.array(current_position) - np.array(last_position)
            )
            if debug:
                print(f"Distance moved: {distance_moved:.3f}m (threshold: 0.05m)")

            if distance_moved < 0.05:  # 5cm 미만으로 이동
                self.stuck_counter += 1
                if debug:
                    print(
                        f"Stuck detected! Counter: {self.stuck_counter}/{self.stuck_threshold}"
                    )
            else:
                if debug and self.stuck_counter > 0:
                    print(
                        f"Movement detected, resetting stuck counter from {self.stuck_counter} to 0"
                    )
                self.stuck_counter = 0

        # LLaVA 서버에 요청
        if self.stuck_counter >= self.stuck_threshold and not self.backtrack_mode:
            if debug:
                print(
                    f"Stuck counter reached threshold ({self.stuck_threshold}), contacting LLaVA server..."
                )
            # vis_dir에서 가장 최근 이미지 가져오기
            try:
                latest_image = self.get_latest_image()
            except Exception as e:
                print(f"Error getting latest image: {e}")
                self.stuck_counter = 0
                self.backtrack_mode = True
                backtrack_x = start_x - 2.0 * math.cos(math.radians(start_o))
                backtrack_y = start_y - 2.0 * math.sin(math.radians(start_o))
                self.backtrack_target = (backtrack_x, backtrack_y)
                if debug:
                    print(f"Reset stuck counter to {self.stuck_counter}")
                    print(
                        f"Fallback to backtrack mode due to image error. Target: {self.backtrack_target}"
                    )
            else:
                # LLaVA 서버로 이미지 전송 및 응답 수신
                self.llava_direction = self.send_image_to_llava(latest_image)
                print(f"LLaVA suggests moving: {self.llava_direction}")

                # LLaVA 방향에 따라 즉시 동작 결정
                if self.llava_direction:
                    self.stuck_counter = 0  # 카운터 초기화
                    self.backtrack_mode = False  # 백트랙킹 모드 비활성화
                    # 방향 초기화
                    if debug:
                        print(
                            f"Reset stuck counter to {self.stuck_counter} after LLaVA communication"
                        )
                    if self.discrete_actions:
                        if self.llava_direction.lower() == "left":
                            return (
                                DiscreteNavigationAction.TURN_LEFT,
                                goal_map,
                                start,
                                obstacle_map,
                            )
                        elif self.llava_direction.lower() == "right":
                            return (
                                DiscreteNavigationAction.TURN_RIGHT,
                                goal_map,
                                start,
                                obstacle_map,
                            )
                        elif self.llava_direction.lower() == "backward":
                            angle_rad = math.radians(start_o)
                            backtrack_x = start_x - 0.5 * math.cos(angle_rad)
                            backtrack_y = start_y - 0.5 * math.sin(angle_rad)
                            self.backtrack_target = (backtrack_x, backtrack_y)
                            self.backtrack_mode = True
                            return (
                                DiscreteNavigationAction.TURN_LEFT,
                                goal_map,
                                start,
                                obstacle_map,
                            )
                    else:
                        if self.llava_direction.lower() == "left":
                            action = ContinuousNavigationAction(
                                [0, 0, math.radians(self.turn_angle)]
                            )
                        elif self.llava_direction.lower() == "right":
                            action = ContinuousNavigationAction(
                                [0, 0, -math.radians(self.turn_angle)]
                            )
                        elif self.llava_direction.lower() == "backward":
                            action = ContinuousNavigationAction(
                                [0, 0, math.radians(180)]
                            )
                            self.backtrack_target = (
                                start_x - 0.5 * math.cos(math.radians(start_o)),
                                start_y - 0.5 * math.sin(math.radians(start_o)),
                            )
                            self.backtrack_mode = True
                        return action, goal_map, start, obstacle_map

        if self.last_action == DiscreteNavigationAction.MOVE_FORWARD or (
            type(self.last_action) == ContinuousNavigationAction
            and np.linalg.norm(self.last_action.xyt[:2]) > 0
        ):
            self._check_collision()
        debug = False
        if self.backtrack_mode:
            backtrack_goal_map = np.zeros_like(goal_map)
            backtrack_x, backtrack_y = self.backtrack_target
            backtrack_cell_x = int(backtrack_y * 100.0 / self.map_resolution - gx1)
            backtrack_cell_y = int(backtrack_x * 100.0 / self.map_resolution - gy1)
            if debug:
                print(f"Backtrack Cell: ({backtrack_cell_x}, {backtrack_cell_y})")
            if (
                0 <= backtrack_cell_x < goal_map.shape[0]
                and 0 <= backtrack_cell_y < goal_map.shape[1]
            ):
                backtrack_goal_map[backtrack_cell_x, backtrack_cell_y] = 1
            target_goal_map = backtrack_goal_map
        else:
            target_goal_map = goal_map

        try:
            (
                short_term_goal,
                closest_goal_map,
                replan,
                stop,
                closest_goal_pt,
                dilated_obstacles,
            ) = self._get_short_term_goal(
                obstacle_map,
                target_goal_map,
                start,
                planning_window,
                plan_to_dilated_goal=use_dilation_for_stg,
                frontier_map=frontier_map,
            )
        except Exception as e:
            print("Warning! Planner crashed with error:", e)
            return (
                DiscreteNavigationAction.STOP,
                np.zeros(goal_map.shape),
                (0, 0),
                np.zeros(goal_map.shape),
            )

        if (
            self.backtrack_mode
            and np.linalg.norm(
                np.array([start_x, start_y]) - np.array([backtrack_x, backtrack_y])
            )
            < 0.2
        ):
            self.backtrack_mode = False
            self.backtrack_target = None
            stuck_x, stuck_y = self.position_history[-1]
            stuck_cell_x = int(stuck_x * 100.0 / self.map_resolution - gx1)
            stuck_cell_y = int(stuck_y * 100.0 / self.map_resolution - gy1)
            self.collision_map[
                stuck_cell_x - 1 : stuck_cell_x + 2, stuck_cell_y - 1 : stuck_cell_y + 2
            ] = 1
            if debug:
                print("Backtrack completed. Added stuck point to collision map.")

        if debug:
            print("Current pose:", start)
            print("Short term goal:", short_term_goal)
            print(
                "  - delta =",
                short_term_goal[0] - start[0],
                short_term_goal[1] - start[1],
            )
            dist_to_short_term_goal = np.linalg.norm(
                start - np.array(short_term_goal[:2])
            )
            print(
                "Distance (m):",
                dist_to_short_term_goal * self.map_resolution * CM_TO_METERS,
            )
            print("Replan:", replan)

        if replan and not stop:
            self.collision_map *= 0
            if self.curr_obs_dilation_selem_radius > self.min_obs_dilation_selem_radius:
                self.curr_obs_dilation_selem_radius -= 1
                self.obs_dilation_selem = skimage.morphology.disk(
                    self.curr_obs_dilation_selem_radius
                )
                if debug:
                    print(
                        f"reduced obs dilation to {self.curr_obs_dilation_selem_radius}"
                    )
            if found_goal:
                if debug:
                    print(
                        "ERROR: Could not find a path to the high-level goal. Trying to explore more..."
                    )
                (
                    short_term_goal,
                    closest_goal_map,
                    replan,
                    stop,
                    closest_goal_pt,
                    dilated_obstacles,
                ) = self._get_short_term_goal(
                    obstacle_map,
                    frontier_map,
                    start,
                    planning_window,
                    plan_to_dilated_goal=True,
                )
                if debug:
                    print("--- after replanning to frontier ---")
                    print("goal =", short_term_goal)
                found_goal = False
                if replan:
                    print("Nowhere left to explore. Stopping.")
                    return (
                        DiscreteNavigationAction.STOP,
                        closest_goal_map,
                        short_term_goal,
                        dilated_obstacles,
                    )

        angle_agent = pu.normalize_angle(start_o)
        stg_x, stg_y = short_term_goal
        relative_stg_x, relative_stg_y = stg_x - start[0], stg_y - start[1]
        angle_st_goal = math.degrees(math.atan2(relative_stg_x, relative_stg_y))
        relative_angle_to_stg = pu.normalize_angle(angle_agent - angle_st_goal)
        goal_x, goal_y = closest_goal_pt
        angle_goal = math.degrees(math.atan2(goal_x - start[0], goal_y - start[1]))
        relative_angle_to_closest_goal = pu.normalize_angle(angle_agent - angle_goal)

        if debug:
            distance_to_goal = np.linalg.norm(np.array([goal_x, goal_y]) - start)
            distance_to_goal_cm = distance_to_goal * self.map_resolution
            print("-----------------")
            print("Found reachable goal:", found_goal)
            print("Stop:", stop)
            print("Angle to goal:", relative_angle_to_closest_goal)
            print("Distance to goal", distance_to_goal)
            print(
                "Distance in cm:", distance_to_goal_cm, ">", self.min_goal_distance_cm
            )
            m_relative_stg_x, m_relative_stg_y = [
                CM_TO_METERS * self.map_resolution * d
                for d in [relative_stg_x, relative_stg_y]
            ]
            print("continuous actions for exploring")
            print("agent angle =", angle_agent)
            print("angle stg goal =", angle_st_goal)
            print("angle final goal =", relative_angle_to_closest_goal)
            print(
                m_relative_stg_x, m_relative_stg_y, "rel ang =", relative_angle_to_stg
            )
            print("-----------------")

        action = self.get_action(
            relative_stg_x,
            relative_stg_y,
            relative_angle_to_stg,
            relative_angle_to_closest_goal,
            start_o,
            found_goal and not self.backtrack_mode,
            stop and not self.backtrack_mode,
            debug,
        )
        self.last_action = action
        return action, closest_goal_map, short_term_goal, dilated_obstacles

    def get_action(
        self,
        relative_stg_x: float,
        relative_stg_y: float,
        relative_angle_to_stg: float,
        relative_angle_to_closest_goal: float,
        start_compass: float,
        found_goal: bool,
        stop: bool,
        debug: bool,
    ):
        """
        Gets discrete/continuous action given short-term goal. Agent orients to closest goal if found_goal=True and stop=True
        """
        # Short-term goal -> deterministic local policy
        if not (found_goal and stop):
            if self.discrete_actions:
                if relative_angle_to_stg > self.turn_angle / 2.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_stg < -self.turn_angle / 2.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    action = DiscreteNavigationAction.MOVE_FORWARD
            else:
                # Use the short-term goal to set where we should be heading next
                m_relative_stg_x, m_relative_stg_y = [
                    CM_TO_METERS * self.map_resolution * d
                    for d in [relative_stg_x, relative_stg_y]
                ]
                if np.abs(relative_angle_to_stg) > self.turn_angle / 2.0:
                    # Must return commands in radians and meters
                    relative_angle_to_stg = math.radians(relative_angle_to_stg)
                    action = ContinuousNavigationAction([0, 0, -relative_angle_to_stg])
                else:
                    # Must return commands in radians and meters
                    relative_angle_to_stg = math.radians(relative_angle_to_stg)
                    xyt_global = [
                        m_relative_stg_y,
                        m_relative_stg_x,
                        -relative_angle_to_stg,
                    ]

                    xyt_local = xyt_global_to_base(
                        xyt_global, [0, 0, math.radians(start_compass)]
                    )
                    xyt_local[
                        2
                    ] = (
                        -relative_angle_to_stg
                    )  # the original angle was already in base frame
                    action = ContinuousNavigationAction(xyt_local)
        else:
            # Try to orient towards the goal object - or at least any point sampled from the goal
            # object.
            if debug:
                print()
                print("----------------------------")
                print(">>> orient towards the goal:", relative_angle_to_closest_goal)
            if self.discrete_actions:
                if relative_angle_to_closest_goal > 2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_closest_goal < -2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    action = DiscreteNavigationAction.STOP
            elif (
                np.abs(relative_angle_to_closest_goal) > self.continuous_angle_tolerance
            ):
                if debug:
                    print("Continuous rotation towards goal point")
                relative_angle_to_closest_goal = math.radians(
                    relative_angle_to_closest_goal
                )
                action = ContinuousNavigationAction(
                    [0, 0, -relative_angle_to_closest_goal]
                )
            else:
                action = DiscreteNavigationAction.STOP
                if debug:
                    print("!!! DONE !!!")

        return action

    def _get_short_term_goal(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        start: List[int],
        planning_window: List[int],
        plan_to_dilated_goal=False,
        frontier_map=None,
        visualize=False,
    ) -> Tuple[Tuple[int, int], np.ndarray, bool, bool]:
        """Get short-term goal.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            start: start location (x, y)
            planning_window: local map boundaries (gx1, gx2, gy1, gy2)
            plan_to_dilated_goal: for objectnav; plans to dialted goal points instead of explicitly checking reach.

        Returns:
            short_term_goal: short-term goal position (x, y) in map
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            replan: binary flag to indicate we couldn't find a plan to reach
             the goal
            stop: binary flag to indicate we've reached the goal
        """
        gx1, gx2, gy1, gy2 = planning_window
        (x1, y1,) = (
            0,
            0,
        )
        x2, y2 = obstacle_map.shape
        obstacles = obstacle_map[x1:x2, y1:y2]

        # Dilate obstacles
        dilated_obstacles = cv2.dilate(obstacles, self.obs_dilation_selem, iterations=1)

        # Create inverse map of obstacles - this is territory we assume is traversible
        # Traversible is now the map
        traversible = 1 - dilated_obstacles
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        agent_rad = self.agent_cell_radius
        traversible[
            int(start[0] - x1) - agent_rad : int(start[0] - x1) + agent_rad + 1,
            int(start[1] - y1) - agent_rad : int(start[1] - y1) + agent_rad + 1,
        ] = 1
        traversible = add_boundary(traversible)
        goal_map = add_boundary(goal_map, value=0)
        planner = FMMPlanner(
            traversible,
            step_size=self.step_size,
            vis_dir=self.vis_dir,
            visualize=self.visualize,
            print_images=self.print_images,
            goal_tolerance=self.goal_tolerance,
        )
        if plan_to_dilated_goal:
            # Compute dilated goal map for use with simulation code - use this to compute closest goal
            dilated_goal_map = cv2.dilate(
                goal_map, self.goal_dilation_selem, iterations=1
            )
            # Set multi goal to the dilated goal map
            # We will now try to find a path to any of these spaces
            self.dd = planner.set_multi_goal(
                dilated_goal_map,
                self.timestep,
                self.dd,
                self.map_downsample_factor,
                self.map_update_frequency,
            )
            goal_distance_map, closest_goal_pt = self.get_closest_traversible_goal(
                traversible, goal_map, start, dilated_goal_map=dilated_goal_map
            )
        else:
            navigable_goal_map = planner._find_within_distance_to_multi_goal(
                goal_map,
                self.min_goal_distance_cm / self.map_resolution,
                timestep=self.timestep,
                vis_dir=self.vis_dir,
            )
            if not np.any(navigable_goal_map):
                frontier_map = add_boundary(frontier_map, value=0)
                navigable_goal_map = frontier_map
            self.dd = planner.set_multi_goal(
                navigable_goal_map,
                self.timestep,
                self.dd,
                self.map_downsample_factor,
                self.map_update_frequency,
            )
            goal_distance_map, closest_goal_pt = self.get_closest_goal(goal_map, start)

        self.timestep += 1

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        # This is where we create the planner to get the trajectory to this state
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(
            state, continuous=(not self.discrete_actions)
        )
        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        short_term_goal = int(stg_x), int(stg_y)

        if visualize:
            print("Start visualizing")
            plt.figure(1)
            plt.subplot(131)
            _navigable_goal_map = navigable_goal_map.copy()
            _navigable_goal_map[int(stg_x), int(stg_y)] = 1
            plt.imshow(np.flipud(_navigable_goal_map))
            plt.plot(stg_x, stg_y, "bx")
            plt.plot(start[0], start[1], "rx")
            plt.subplot(132)
            plt.imshow(np.flipud(planner.fmm_dist))
            plt.subplot(133)
            plt.imshow(np.flipud(planner.traversible))
            plt.show()
            print("Done visualizing.")

        return (
            short_term_goal,
            goal_distance_map,
            replan,
            stop,
            closest_goal_pt,
            dilated_obstacles,
        )

    def get_closest_traversible_goal(
        self, traversible, goal_map, start, dilated_goal_map=None
    ):
        """Old version of the get_closest_goal function, which takes into account the distance along geometry to a goal object. This will tell us the closest point on the goal map, both for visualization and for orienting towards it to grasp. Uses traversible to sort this out."""

        # NOTE: this is the old version - before adding goal dilation
        # vis_planner = FMMPlanner(traversible)
        # TODO How to do this without the overhead of creating another FMM planner?
        traversible_ = traversible.copy()
        if dilated_goal_map is None:
            traversible_[goal_map == 1] = 1
        else:
            traversible_[dilated_goal_map == 1] = 1
        vis_planner = FMMPlanner(traversible_)
        curr_loc_map = np.zeros_like(goal_map)
        # Update our location for finding the closest goal
        curr_loc_map[start[0], start[1]] = 1
        # curr_loc_map[short_term_goal[0], short_term_goal]1]] = 1
        vis_planner.set_multi_goal(curr_loc_map)
        fmm_dist_ = vis_planner.fmm_dist.copy()
        # find closest point on non-dilated goal map
        goal_map_ = goal_map.copy()
        goal_map_[goal_map_ == 0] = 10000
        fmm_dist_[fmm_dist_ == 0] = 10000
        closest_goal_map = (goal_map_ * fmm_dist_) == (goal_map_ * fmm_dist_).min()
        closest_goal_map = remove_boundary(closest_goal_map)
        closest_goal_pt = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        return closest_goal_map, closest_goal_pt

    def get_closest_goal(self, goal_map, start):
        """closest goal, avoiding any obstacles."""
        empty = np.ones_like(goal_map)
        empty_planner = FMMPlanner(empty)
        empty_planner.set_goal(start)
        dist_map = empty_planner.fmm_dist * goal_map
        dist_map[dist_map == 0] = 10000
        closest_goal_map = dist_map == dist_map.min()
        closest_goal_map = remove_boundary(closest_goal_map)
        closest_goal_pt = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        return closest_goal_map, closest_goal_pt

    def _check_collision(self):
        """Check whether we had a collision and update the collision map."""
        x1, y1, t1 = self.last_pose
        x2, y2, _ = self.curr_pose
        buf = 4
        length = 2

        # You must move at least 5 cm when doing forward actions
        # Otherwise we assume there has been a collision
        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.col_width += 2
            if self.col_width == 7:
                length = 4
                buf = 3
            self.col_width = min(self.col_width, 5)
        else:
            self.col_width = 1

        dist = pu.get_l2_distance(x1, x2, y1, y2)

        if dist < self.collision_threshold:
            # We have a collision
            width = self.col_width

            # Add obstacles to the collision map
            for i in range(length):
                for j in range(width):
                    wx = x1 + 0.05 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + 0.05 * (
                        (i + buf) * np.sin(np.deg2rad(t1))
                        - (j - width // 2) * np.cos(np.deg2rad(t1))
                    )
                    r, c = wy, wx
                    r, c = int(r * 100 / self.map_resolution), int(
                        c * 100 / self.map_resolution
                    )
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r, c] = 1

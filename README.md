## 🛠 Custom Modifications for Scene-based Visualization

이 프로젝트에서는 `home-robot` 시뮬레이터에 **scene 단위 이미지 저장 및 관리 기능**을 추가했습니다.  
아래는 기능별 설명과 관련 코드 위치입니다.

---

### 📂 이미지 저장 기능

#### ✅ 이미지 저장을 scene별로 분리
- 이미지 저장 디렉토리는 `scene_id` 기준으로 설정됩니다.
- 관련 함수:
  - `src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/visualizer.py`
    - `set_vis_dir(scene_id)` 함수 수정

#### ✅ scene 변경 시 구분
- `reset()` 시마다 새로운 `scene_id`에 따라 이미지가 저장되므로,
  서로 다른 scene의 이미지가 섞이지 않도록 구분됨

---

### 🧹 scene 전환 시 이미지 자동 제거 기능 (※ 주석처리 상태)

- scene이 바뀔 때 이전 scene의 이미지 디렉토리를 삭제하는 `rm_folder()` 함수가 구현되어 있습니다.
- 현재는 **안전성을 위해 주석처리** 되어 있으며, 필요 시 주석을 해제하여 사용 가능합니다.

#### 🔧 구현 위치
- `src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/visualizer.py`
  - `def rm_folder(dir_path)` 함수 추가
- `src/home_robot_sim/home_robot_sim/env/habitat_ovmm_env/habitat_ovmm_env.py`
  - `def rm_folder(dir_path)` 함수 동일하게 추가
  - `reset()` 함수 내부에서 scene 전환 감지 가능

---

### 🧪 평가 코드 변경

#### 📍 평가 루프에서 이미지 저장 디렉토리 초기화 연결

- `projects/habitat_ovmm/evaluator.py`
  - `local_evaluate()` 함수 내부에서 `self.env.reset()`이 호출되며,
    그 과정에서 `set_vis_dir()`이 호출되어 scene별 디렉토리를 자동 설정합니다.
  - `num_episodes()` 함수 내부에서 몇 개의 scene이 사용되는지 제어 가능

---

### 🗂 Scene 데이터 경로

```bash
data/hssd-hab/scenes-uncluttered/{scene_id}.scene_instance.json

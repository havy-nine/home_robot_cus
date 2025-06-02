import glob
import os

import cv2

# 이미지 폴더 경로 설정
folder_path = "/home/intern/cheon/home-robot/datadump_sam_vlm45/images/eval_hssd_sam/107734176_176000019_48"

# 출력 영상 파일 이름과 설정
output_video = "cropped_output_video.mp4"
fps = 30  # 초당 프레임 수

# 크롭할 좌표 설정 (x_min, y_min, x_max, y_max)
x_min, y_min, x_max, y_max = 1300, 50, 1750, 520
crop_width = x_max - x_min
crop_height = y_max - y_min

# 이미지 파일 목록 가져오기
image_files = sorted(
    glob.glob(os.path.join(folder_path, "snapshot_*.png"))
)  # 확장자 수정 필요 시 변경

# 비디오 라이터 설정
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 코덱 설정
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (crop_width, crop_height))

# 모든 이미지를 읽어서 크롭 후 영상에 추가
for image_file in image_files:
    frame = cv2.imread(image_file)
    # 이미지 크롭
    cropped_frame = frame[y_min:y_max, x_min:x_max]
    video_writer.write(cropped_frame)

# 작업 완료 후 리소스 해제
video_writer.release()
print(f"크롭된 영상이 생성되었습니다: {output_video}")

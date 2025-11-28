import cv2, os

save_dir = "../../lane_project/datasets/train/images/"  # ← 既存のフォルダ

# フォルダが存在するかチェック（存在しないとエラーになる）
if not os.path.isdir(save_dir):
    raise Exception(f"Directory does not exist: {save_dir}")

cap = cv2.VideoCapture("./datasets_video/line_video.mp4")

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if i % 5 == 0:
        cv2.imwrite(f"{save_dir}/{i:06d}.jpg", frame)
    i += 1

cap.release()
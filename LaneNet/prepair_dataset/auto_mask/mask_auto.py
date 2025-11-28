# mask_auto.py : 簡易自動ラベリング（白ライン想定）
import cv2, os, glob
import numpy as np

src_dir = "./lane_project/datasets/train/images"
out_bin = "./lane_project/datasets/train/binary_masks"
out_inst = "./lane_priject/datasets/train/instance_masks"
os.makedirs(out_bin, exist_ok=True); os.makedirs(out_inst, exist_ok=True)

for img_path in glob.glob(os.path.join(src_dir, "*.jpg")):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 黄色の範囲（必要なら後で調整）
    lower = np.array([15, 80, 80])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # ノイズ除去
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 保存
    name = os.path.basename(img_path)
    cv2.imwrite(f"dataset/train/binary_labels/{name}", mask)
    cv2.imwrite(f"dataset/train/instance_labels/{name}", (mask > 0).astype('uint8') * 255)
    
    print("yay")

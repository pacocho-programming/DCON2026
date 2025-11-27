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
    # 白の範囲（要調整）
    lower = np.array([0,0,180])
    upper = np.array([180,40,255])
    mask = cv2.inRange(hsv, lower, upper)
    # ノイズ除去
    kern = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    # 保存 (binary mask)
    fname = os.path.basename(img_path).replace(".jpg", ".png")
    cv2.imwrite(os.path.join(out_bin, fname), mask)
    # instance mask: 1 where mask>0, else 0
    inst = (mask>0).astype(np.uint8) * 1
    cv2.imwrite(os.path.join(out_inst, fname), inst)
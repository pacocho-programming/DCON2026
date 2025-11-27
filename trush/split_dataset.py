import os, shutil, random, glob

# パス設定
img_train_path = 'dataset/images/train'
lbl_train_path = 'dataset/labels/train'
img_val_path = 'dataset/images/val'
lbl_val_path = 'dataset/labels/val'

os.makedirs(img_val_path, exist_ok=True)
os.makedirs(lbl_val_path, exist_ok=True)

# 画像一覧取得
images = glob.glob(img_train_path + '/*.jpg') + glob.glob(img_train_path + '/*.jpeg') + glob.glob(img_train_path + '/*.png')
random.shuffle(images)

# 80% train, 20% val
split_idx = int(len(images) * 0.8)
val_images = images[split_idx:]

for img_path in val_images:
    label_path = img_path.replace('images/train', 'labels/train').rsplit('.', 1)[0] + '.txt'
    shutil.move(img_path, img_path.replace('images/train', 'images/val'))
    shutil.move(label_path, label_path.replace('labels/train', 'labels/val'))

print(f"Train: {split_idx}, Val: {len(val_images)}")


import pandas as pd
import os

def rename_gt_images(csv_path):
    df = pd.read_csv(csv_path, sep='\t')
    image_paths = df['filepath'].tolist()
    image_paths = [p.replace("/workspace/datasets/SD_Rest", "/data/coding") for p in image_paths]

    for lq_path in image_paths:
        # 解析降解类型
        parts = lq_path.split(os.sep)
        # 假设路径结构是 .../train/[degration_type]/LQ/[filename]
        if len(parts) < 4 or parts[-3] not in ['shadowed', 'raindrop']:
            continue # 确保路径结构正确且降解类型是目标之一
        degration_type = parts[-3]

        lq_filename = os.path.basename(lq_path)
        gt_dir = os.path.join(os.path.dirname(os.path.dirname(lq_path)), 'GT')

        # 原GT文件名生成
        if degration_type == 'shadowed':
            base, ext = os.path.splitext(lq_filename)
            original_gt_filename = f"{base}_no_shadow{ext}"
            original_gt_filename = f"{base}_free{ext}"
        elif degration_type == 'raindrop':
            original_gt_filename = lq_filename.replace('_rain', '_clean')
        else:
            continue # 其他类型跳过

        original_gt_path = os.path.join(gt_dir, original_gt_filename)
        new_gt_path = os.path.join(gt_dir, lq_filename)

        # 检查原GT文件是否存在
        if os.path.exists(original_gt_path):
            # 重命名
            os.rename(original_gt_path, new_gt_path)
            print(f"Renamed {original_gt_path} to {new_gt_path}")
        else:
            print(f"Warning: {original_gt_path} does not exist")

# rename_gt_images('/data/coding/daclip_train.csv')
rename_gt_images('/data/coding/daclip_val.csv')
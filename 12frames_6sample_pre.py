#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成训练数据：每个视频生成6个样本，每样本12帧
- 总样本数 = 300 × 6 = 1800
- 训练集 1440，验证集 360
"""

import json
import os
import re
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# ==================== 路径配置 ====================
FRAMES_DIR = r"F:\download\surgery_frames"
SEMANTICS_FILE = r"F:\download\qwen\Qwen3-VL-main\qwen-vl-finetune\qwenvl\data\annotations\semantics_labeled.txt"
MOTION_FILE = r"F:\download\qwen\Qwen3-VL-main\qwen-vl-finetune\qwenvl\data\annotations\motion_changes_final.txt"
LABELS_FILE = r"F:\download\TASK2\laparoscope_motion_label.txt"
OUTPUT_DIR = r"F:\download"

# ==================== 6种样本定义 ====================
SAMPLES = [
    # 样本1: 4个语义（S1,S2,S5,S6）
    {
        "frames": [1, 5, 10, 25, 35, 53, 70, 78, 95, 100, 115, 125],
        "semantic_frames": [
            {"frame": 1, "semantic_suffix": "001"},   # S1
            {"frame": 25, "semantic_suffix": "025"},  # S2
            {"frame": 100, "semantic_suffix": "100"}, # S5
            {"frame": 125, "semantic_suffix": "125"}, # S6
        ]
    },
    # 样本2: 4个语义（S2,S3,S4,S5）
    {
        "frames": [4, 7, 25, 28, 40, 50, 65, 75, 85, 100, 120, 128],
        "semantic_frames": [
            {"frame": 25, "semantic_suffix": "025"},  # S2
            {"frame": 50, "semantic_suffix": "050"},  # S3
            {"frame": 85, "semantic_suffix": "085"},  # S4
            {"frame": 100, "semantic_suffix": "100"}, # S5
        ]
    },
    # 样本3: 3个语义（S1,S3,S5）
    {
        "frames": [1, 7, 28, 31, 45, 50, 70, 78, 90, 100, 110, 131],
        "semantic_frames": [
            {"frame": 1, "semantic_suffix": "001"},   # S1
            {"frame": 50, "semantic_suffix": "050"},  # S3
            {"frame": 100, "semantic_suffix": "100"}, # S5
        ]
    },
    # 样本4: 3个语义（S2,S4,S5）
    {
        "frames": [5, 15, 25, 34, 40, 53, 65, 75, 84, 100, 115, 134],
        "semantic_frames": [
            {"frame": 25, "semantic_suffix": "025"},  # S2
            {"frame": 75, "semantic_suffix": "075"},  # S4
            {"frame": 100, "semantic_suffix": "100"}, # S5
        ]
    },
    # 样本5: 4个语义（S1,S4,S5,S6）
    {
        "frames": [1, 7, 20, 28, 35, 56, 70, 75, 95, 100, 120, 125],
        "semantic_frames": [
            {"frame": 1, "semantic_suffix": "001"},   # S1
            {"frame": 75, "semantic_suffix": "075"},  # S4
            {"frame": 100, "semantic_suffix": "100"}, # S5
            {"frame": 125, "semantic_suffix": "125"}, # S6
        ]
    },
    # 样本6: 3个语义（S3,S5,S6）
    {
        "frames": [4, 10, 20, 31, 34, 50, 60, 81, 90, 100, 110, 125],
        "semantic_frames": [
            {"frame": 50, "semantic_suffix": "050"},  # S3
            {"frame": 100, "semantic_suffix": "100"}, # S5
            {"frame": 125, "semantic_suffix": "125"}, # S6
        ]
    },
]

# ================================================

def parse_semantics(path):
    """解析语义描述文件"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^\*\*(\d+):\*\*\s*(.+)$', line)
            if match:
                img_id = match.group(1)
                video_id = img_id[:3]
                frame_suffix = img_id[3:]
                if video_id not in data:
                    data[video_id] = {}
                data[video_id][frame_suffix] = match.group(2).strip()

    print(f"✅ 解析 {len(data)} 个视频的语义描述")
    return data

def parse_motion_changes(path):
    """解析运动变化文件"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^(\d+):\s*(.+)$', line)
            if match:
                video_id = match.group(1).zfill(3)
                motion_desc = match.group(2).strip()
                data[video_id] = motion_desc

    print(f"✅ 解析 {len(data)} 个视频的运动变化")
    return data

def parse_labels(path):
    """解析标签文件"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                clip_id = parts[0].strip().zfill(3)
                try:
                    label = int(parts[1])
                    data[clip_id] = label
                except ValueError:
                    continue

    print(f"✅ 解析 {len(data)} 个视频标签")
    return data

def get_frame_suffix(frame_num):
    """将帧号转换为3位数字符串"""
    return f"{frame_num:03d}"

def create_sample(video_id, sample_config, semantics_dict, motion_desc, label, frames_dir):
    """根据配置创建单个训练样本"""
    frame_numbers = sample_config["frames"]
    semantic_frames = sample_config["semantic_frames"]

    # 构建语义映射
    semantic_map = {}
    for sf in semantic_frames:
        frame = sf["frame"]
        suffix = sf["semantic_suffix"]
        semantic_map[frame] = suffix

    # 生成图片文件名列表
    image_paths = []
    frame_infos = []

    for fn in frame_numbers:
        suffix = get_frame_suffix(fn)
        img_id = f"{video_id}{suffix}"
        img_filename = f"{img_id}.jpg"
        img_path = os.path.join(frames_dir, img_filename)
        if not os.path.exists(img_path):
            return None
        image_paths.append(img_filename)

        # 记录帧信息
        if fn in semantic_map:
            sem_suffix = semantic_map[fn]
            if sem_suffix in semantics_dict:
                semantic_text = semantics_dict[sem_suffix]
            else:
                semantic_text = None
        else:
            semantic_text = None

        frame_infos.append({
            "frame_num": fn,
            "has_semantic": fn in semantic_map,
            "semantic": semantic_text
        })

    if len(image_paths) != 12:
        return None

    # 构建 Human 指令
    image_tags = "<image>\n" * 12

    # 构建帧描述文本
    frame_descriptions = []
    for i, info in enumerate(frame_infos):
        frame_num = info["frame_num"]
        if info["has_semantic"] and info["semantic"]:
            frame_descriptions.append(f"Frame {i+1} (frame {frame_num}): {info['semantic']}")
        else:
            frame_descriptions.append(f"Frame {i+1} (frame {frame_num}): No semantic description")

    sem_content = "\n".join(frame_descriptions)

    # 运动变化信息
    motion_info = f"\n\nMotion analysis: {motion_desc}" if motion_desc else ""

    human_value = (
        f"{image_tags}"
        f"These are 12 frames (frames {', '.join([str(f) for f in frame_numbers])}) from a surgery video:\n\n"
        f"{sem_content}"
        f"{motion_info}\n\n"
        f"Based on these 12 frames and the motion analysis, predict the laparoscope motion at the next frame.\n"
        f"Options: 0:static, 1:up, 2:down, 3:left, 4:right, 5:zoom-in, 6:zoom-out\n"
        f"Output only the number (0-6)."
    )

    return {
        "image": image_paths,
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": str(label)}
        ]
    }

def main():
    print("=" * 80)
    print("🚀 生成训练数据（每个视频生成6个样本，每样本12帧）")
    print("=" * 80)

    # 解析数据
    semantics = parse_semantics(SEMANTICS_FILE)
    motion_changes = parse_motion_changes(MOTION_FILE)
    labels = parse_labels(LABELS_FILE)

    print(f"\n📊 原始数据统计:")
    print(f"   总视频数: {len(labels)}")
    print(f"   有语义的视频: {len(semantics)}")
    print(f"   有运动变化的视频: {len(motion_changes)}")

    # 收集所有样本
    all_samples = []
    sample_labels = []

    skipped_videos = 0

    for video_id, label in labels.items():
        # 检查是否有语义
        if video_id not in semantics:
            skipped_videos += 1
            continue

        semantics_dict = semantics[video_id]
        motion_desc = motion_changes.get(video_id, "")

        # 为每个样本配置生成样本
        for sample_config in SAMPLES:
            sample = create_sample(
                video_id,
                sample_config,
                semantics_dict,
                motion_desc,
                label,
                FRAMES_DIR
            )
            if sample:
                all_samples.append(sample)
                sample_labels.append(label)

    print(f"\n✅ 共创建 {len(all_samples)} 个有效样本")
    print(f"   (期望: {len(labels)}视频 × 6样本 = {len(labels)*6}样本)")
    print(f"   跳过视频数: {skipped_videos}")

    if len(all_samples) == 0:
        print("❌ 错误：没有有效的训练样本")
        return

    # 分层抽样（按标签划分训练/验证集）
    train_indices, val_indices = train_test_split(
        range(len(all_samples)),
        test_size=0.2,
        random_state=42,
        stratify=sample_labels
    )

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    # 保存文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_file = os.path.join(OUTPUT_DIR, 'surgery_train_12frames_6samples.json')
    val_file = os.path.join(OUTPUT_DIR, 'surgery_val_12frames_6samples.json')

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)

    print(f"\n📁 训练集：{len(train_samples)} 条 → {train_file}")
    print(f"📁 验证集：{len(val_samples)} 条 → {val_file}")

    # 标签分布统计
    print("\n" + "=" * 80)
    print("📊 标签分布统计")
    print("=" * 80)

    label_names = ['0:static', '1:up', '2:down', '3:left', '4:right', '5:zoom-in', '6:zoom-out']

    train_labels = [sample_labels[i] for i in train_indices]
    val_labels = [sample_labels[i] for i in val_indices]

    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    total_counter = Counter(sample_labels)

    print(f"\n{'Label':<15} | {'Train':<8} | {'Val':<8} | {'Total':<8} | {'比例':<8}")
    print("-" * 55)

    for i in range(7):
        t_count = train_counter.get(i, 0)
        v_count = val_counter.get(i, 0)
        tot_count = total_counter.get(i, 0)
        ratio = tot_count / len(all_samples) * 100 if len(all_samples) > 0 else 0
        print(f"{label_names[i]:<15} | {t_count:<8} | {v_count:<8} | {tot_count:<8} | {ratio:.1f}%")

    print(f"\n{'总计':<15} | {len(train_samples):<8} | {len(val_samples):<8} | {len(all_samples):<8} | 100%")

    print(f"\n📝 输出文件:")
    print(f"   训练集: {train_file}")
    print(f"   验证集: {val_file}")

    # 显示一个样本示例
    if len(train_samples) > 0:
        print("\n📋 样本示例:")
        sample = train_samples[0]
        print(f"  图片数量: {len(sample['image'])}")
        print(f"  图片示例: {sample['image'][:3]}...")
        print(f"  标签: {sample['conversations'][1]['value']}")

if __name__ == "__main__":
    main()
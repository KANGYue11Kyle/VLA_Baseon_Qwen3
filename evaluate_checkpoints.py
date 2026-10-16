#!/usr/bin/env python
# -*- coding: utf-8 -*-
# eval_fixed_optimized.py

import json
import torch
import re
import os
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
import time
from collections import defaultdict

# ==================== 配置 ====================
BASE_MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-4B-Instruct"
CHECKPOINT_DIR = "/root/autodl-tmp/Qwen3-VL-main/qwen-vl-finetune/qwenvl/train/checkpoints/surgery_final_v5"
VAL_JSON_PATH = "/root/autodl-tmp/Qwen3-VL-main/qwen-vl-finetune/qwenvl/data/surgery_val_final.json"
IMAGE_ROOT = "/root/autodl-tmp/Qwen3-VL-main/qwen-vl-finetune/qwenvl/data/surgery_frames"
TRAINER_STATE_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint-*/trainer_state.json")  # 训练日志路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 16
BATCH_SIZE = 4  # 批处理大小
CHECKPOINT_STEP = 50  # 改为50步
# ============================================

def extract_label(text):
    """提取标签"""
    match = re.search(r'\b([0-6])\b', text.strip())
    return int(match.group(1)) if match else -1

def load_training_loss(checkpoint_path):
    """从训练日志中读取真实loss"""
    try:
        # 查找trainer_state.json文件
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r', encoding='utf-8') as f:
                trainer_state = json.load(f)
            
            # 获取最后一步的loss
            if 'log_history' in trainer_state and trainer_state['log_history']:
                # 找最新的loss记录
                for log in reversed(trainer_state['log_history']):
                    if 'loss' in log:
                        return log['loss']
                    elif 'eval_loss' in log:
                        return log['eval_loss']
            
            # 如果从log_history找不到，尝试从其他字段获取
            if 'last_log' in trainer_state and 'loss' in trainer_state['last_log']:
                return trainer_state['last_log']['loss']
                
        # 尝试读取training_args.bin中的信息
        training_args_path = os.path.join(checkpoint_path, "training_args.bin")
        if os.path.exists(training_args_path):
            # 这是一个二进制文件，可能需要用torch.load读取
            try:
                training_args = torch.load(training_args_path)
                if hasattr(training_args, 'logging_dir'):
                    # 尝试从logging_dir读取loss
                    pass
            except:
                pass
                
    except Exception as e:
        print(f"读取loss失败: {e}")
    
    return None

def load_and_preprocess_data(val_data, image_root):
    """预加载和预处理所有验证数据"""
    print("预处理验证数据...")
    processed_samples = []
    invalid_count = 0
    
    for idx, sample in enumerate(tqdm(val_data, desc="预处理")):
        try:
            image_files = sample['image']
            instruction = sample['conversations'][0]['value']
            true_label = int(sample['conversations'][1]['value'].strip())
            
            # 检查图片是否存在
            valid = True
            for img_file in image_files:
                img_path = os.path.join(image_root, img_file)
                if not os.path.exists(img_path):
                    valid = False
                    break
            
            if valid and len(image_files) == 6:  # 确保有6张图片
                # 预加载图片路径而不是图片本身，节省内存
                img_paths = [os.path.join(image_root, img_file) for img_file in image_files]
                processed_samples.append({
                    'img_paths': img_paths,
                    'instruction': instruction,
                    'true_label': true_label,
                    'idx': idx
                })
            else:
                invalid_count += 1
                
        except Exception as e:
            invalid_count += 1
            continue
    
    print(f"有效样本数: {len(processed_samples)}, 无效样本数: {invalid_count}")
    return processed_samples

def load_images_batch(img_paths_list):
    """批量加载图像"""
    batch_images = []
    for img_paths in img_paths_list:
        images = []
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        batch_images.append(images)
    return batch_images

def evaluate_checkpoint_efficient(checkpoint_path, val_samples, processor, device, batch_size=BATCH_SIZE):
    """高效评估单个checkpoint"""
    print(f"\n{'='*60}")
    print(f"评估: {os.path.basename(checkpoint_path)}")
    print('='*60)
    
    # 读取真实loss
    loss_value = load_training_loss(checkpoint_path)
    if loss_value is not None:
        print(f"训练Loss: {loss_value:.4f}")
    else:
        print("训练Loss: 未找到")
    
    # 加载模型
    print("加载模型...")
    start_time = time.time()
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    
    # 可选：启用模型编译加速（PyTorch 2.0+）
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        print("启用模型编译加速...")
        model = torch.compile(model, mode="reduce-overhead")
    
    load_time = time.time() - start_time
    print(f"模型加载时间: {load_time:.2f}秒")
    
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    # 批处理评估
    print(f"开始批处理评估 (batch_size={batch_size})...")
    eval_start_time = time.time()
    
    for i in tqdm(range(0, len(val_samples), batch_size), desc="评估批次"):
        batch_samples = val_samples[i:i+batch_size]
        
        try:
            # 批量加载图像
            batch_img_paths = [sample['img_paths'] for sample in batch_samples]
            batch_images = load_images_batch(batch_img_paths)
            
            # 批量构建消息和输入
            batch_inputs = []
            for j, sample in enumerate(batch_samples):
                # 构建消息
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img} for img in batch_images[j]
                    ] + [{"type": "text", "text": sample['instruction']}]
                }]
                
                # 处理输入
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=text,
                    images=batch_images[j],
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                batch_inputs.append((inputs, sample))
            
            # 批量生成
            with torch.no_grad():
                for inputs, sample in batch_inputs:
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
                    
                    # 解码
                    generated_text = processor.decode(
                        generated_ids[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    
                    pred_label = extract_label(generated_text)
                    true_label = sample['true_label']
                    
                    if pred_label != -1:
                        total += 1
                        class_total[true_label] += 1
                        if pred_label == true_label:
                            correct += 1
                            class_correct[true_label] += 1
                        
                        # 打印前几个样本
                        if total <= 10:
                            print(f"\n  样本{total}: 真实={true_label}, 预测={pred_label}, 输出='{generated_text}'")
        
        except Exception as e:
            print(f"批次处理错误: {e}")
            continue
    
    eval_time = time.time() - eval_start_time
    acc = correct / total if total > 0 else 0
    
    print(f"\n✅ 准确率: {acc:.4f} ({correct}/{total})")
    print(f"✅ 成功处理样本数: {total}")
    print(f"✅ 评估时间: {eval_time:.2f}秒")
    if total > 0:
        print(f"✅ 平均每样本时间: {eval_time/total:.3f}秒")
    
    # 显示各类别准确率
    if total > 0:
        print("\n各类别准确率:")
        action_names = {0:"静止", 1:"向上", 2:"向下", 3:"向左", 4:"向右", 5:"放大", 6:"缩小"}
        for label in range(7):
            if class_total[label] > 0:
                class_acc = class_correct[label] / class_total[label]
                print(f"  {action_names[label]}({label}): {class_acc:.2%} ({class_correct[label]}/{class_total[label]})")
    
    # 清理内存
    del model
    torch.cuda.empty_cache()
    
    return acc, correct, total, eval_time, loss_value

def main():
    """主函数"""
    # 预加载验证数据
    with open(VAL_JSON_PATH, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"原始验证集样本数: {len(val_data)}")
    
    # 预处理数据
    val_samples = load_and_preprocess_data(val_data, IMAGE_ROOT)
    
    # 收集checkpoint（每50步）
    checkpoints = []
    for d in os.listdir(CHECKPOINT_DIR):
        if d.startswith('checkpoint-'):
            step = int(d.split('-')[1])
            if step % CHECKPOINT_STEP == 0:
                checkpoints.append((step, os.path.join(CHECKPOINT_DIR, d)))
    
    checkpoints.sort(key=lambda x: x[0])
    print(f"\n找到 {len(checkpoints)} 个checkpoint (每{CHECKPOINT_STEP}步一个)")
    if checkpoints:
        print(f"步数范围: {checkpoints[0][0]} - {checkpoints[-1][0]}")
    
    # 加载处理器（所有checkpoint共享）
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    
    print("\n" + "=" * 110)
    print(f"{'Step':<10} {'Accuracy':<12} {'Correct':<8} {'Total':<8} {'Loss':<10} {'Time(s)':<10}")
    print("-" * 110)
    
    results = []
    for step, path in checkpoints:
        acc, cor, tot, eval_time, loss_val = evaluate_checkpoint_efficient(
            path, val_samples, processor, DEVICE, BATCH_SIZE
        )
        
        loss_display = f"{loss_val:.4f}" if loss_val is not None else "N/A"
        print(f"{step:<10} {acc:<12.4f} {cor:<8} {tot:<8} {loss_display:<10} {eval_time:<10.2f}")
        
        results.append({
            "step": step, 
            "accuracy": acc,
            "loss": loss_val,
            "correct": cor,
            "total": tot,
            "eval_time": eval_time
        })
    
    # 找出最佳checkpoint
    if results:
        # 根据准确率找最佳
        best_by_acc = max(results, key=lambda x: x['accuracy'])
        print("\n" + "=" * 110)
        print(f"🏆 最佳模型(按准确率): Step {best_by_acc['step']}, 准确率: {best_by_acc['accuracy']:.4f}")
        
        # 如果有loss值，也按loss找最佳
        valid_loss_results = [r for r in results if r['loss'] is not None]
        if valid_loss_results:
            best_by_loss = min(valid_loss_results, key=lambda x: x['loss'])
            print(f"🏆 最佳模型(按loss): Step {best_by_loss['step']}, Loss: {best_by_loss['loss']:.4f}")
        print("=" * 110)
    
    # 保存结果
    output_file = 'checkpoint_evaluation_results_50.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_file}")
    
    # 输出统计信息
    print("\n统计信息:")
    print(f"评估的checkpoint总数: {len(results)}")
    if results:
        accuracies = [r['accuracy'] for r in results]
        print(f"准确率范围: {min(accuracies):.4f} - {max(accuracies):.4f}")
        print(f"平均准确率: {sum(accuracies)/len(accuracies):.4f}")

if __name__ == "__main__":
    main()
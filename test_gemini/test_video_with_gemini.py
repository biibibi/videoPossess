#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_video_with_gemini.py - 使用Gemini API分析视频

import asyncio
import time
import cv2
import os
import base64
import json
from pathlib import Path
from main import ModelProcessor

async def analyze_video(video_path, output_dir="results", targets=None, frame_interval=30):
    """
    使用Gemini API分析视频文件的每隔N帧图像
    
    Args:
        video_path: 视频文件路径
        output_dir: 结果输出目录
        targets: 要检测的目标列表，如 ["人物", "电脑", "桌子"]
        frame_interval: 帧间隔，每隔多少帧分析一次
    """
    if targets is None:
        targets = ["人物", "电脑", "文件", "手机", "桌子", "椅子"]
    
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    frames_dir = output_path / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"视频信息:")
    print(f"  - 路径: {video_path}")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - 时长: {duration:.2f}秒")
    print(f"将每隔 {frame_interval} 帧分析一次，约每 {frame_interval/fps:.2f} 秒一次")
    
    # 分析结果列表
    results = []
    frame_count = 0
    analyzed_count = 0
    
    print("\n开始分析视频...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔frame_interval帧分析一次
        if frame_count % frame_interval == 0:
            frame_timestamp = frame_count / fps
            print(f"分析第 {frame_count} 帧 (时间点: {frame_timestamp:.2f}秒)...")
            
            # 保存帧为图像文件
            frame_file = frames_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            
            # 将帧转换为base64编码
            success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                print(f"  - 警告: 无法编码第 {frame_count} 帧")
                frame_count += 1
                continue
                
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 使用Gemini API分析帧
            api_start = time.time()
            api_result = await ModelProcessor.process(
                model_name="gemini", 
                image=image_base64, 
                targets=targets, 
                start_time=api_start
            )
            api_duration = time.time() - api_start
            
            # 添加帧信息
            api_result["frame_index"] = frame_count
            api_result["frame_timestamp"] = frame_timestamp
            api_result["frame_file"] = str(frame_file)
            api_result["api_duration"] = api_duration
            
            # 打印分析结果摘要
            found_targets = [t["name"] for t in api_result.get("targets", []) if t.get("found")]
            print(f"  - 分析完成 ({api_duration:.2f}秒)")
            print(f"  - 描述: {api_result.get('description', '')[:100]}..." if len(api_result.get('description', '')) > 100 
                  else f"  - 描述: {api_result.get('description', '')}")
            print(f"  - 找到目标: {found_targets}")
            
            # 保存分析结果
            results.append(api_result)
            analyzed_count += 1
            
            # 定期保存中间结果到JSON文件
            if analyzed_count % 5 == 0:
                interim_file = output_path / "interim_results.json"
                with open(interim_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"已保存中间结果到 {interim_file}")
            
        frame_count += 1
    
    # 释放视频资源
    cap.release()
    
    # 计算总用时
    total_duration = time.time() - start_time
    
    # 保存分析结果到JSON文件
    results_file = output_path / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n分析完成:")
    print(f"  - 分析了 {analyzed_count}/{total_frames} 帧")
    print(f"  - 总耗时: {total_duration:.2f}秒")
    print(f"  - 结果保存到: {results_file}")
    print(f"  - 帧图像保存到: {frames_dir}")
    
    return results

async def main():
    """主函数"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='使用Gemini API分析视频')
    parser.add_argument('video_path', help='视频文件路径')
    parser.add_argument('--targets', nargs='+', help='要检测的目标，如 --targets 人物 电脑 桌子')
    parser.add_argument('--interval', type=int, default=30, help='帧间隔，每隔多少帧分析一次，默认30')
    parser.add_argument('--output', default='results', help='结果输出目录，默认为results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"错误: 视频文件 {args.video_path} 不存在")
        return 1
    
    await analyze_video(
        video_path=args.video_path,
        output_dir=args.output,
        targets=args.targets,
        frame_interval=args.interval
    )
    
    return 0

if __name__ == "__main__":
    asyncio.run(main()) 
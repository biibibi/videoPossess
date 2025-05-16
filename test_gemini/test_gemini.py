#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_gemini.py - 测试Google Gemini集成

import base64
import os
import sys
import google.generativeai as genai  # 修正导入方式
from config import GEMINI_CONFIG, MODELS
import asyncio
import time

async def test_gemini_integration():
    """测试Gemini API集成是否正常工作"""
    print("开始测试Gemini API集成...")
    
    # 从配置获取API密钥和代理设置
    gemini_config = MODELS["gemini"]
    api_key = gemini_config["api_key"]
    model_name = gemini_config.get("name", "gemini-2.0-flash-exp")
    proxy = GEMINI_CONFIG.get("proxy", {})
    
    # 设置代理
    if proxy.get("http"):
        os.environ["HTTP_PROXY"] = proxy.get("http")
    if proxy.get("https"):
        os.environ["HTTPS_PROXY"] = proxy.get("https")
    
    print(f"使用模型: {model_name}")
    print(f"HTTP代理: {os.environ.get('HTTP_PROXY', '无')}")
    print(f"HTTPS代理: {os.environ.get('HTTPS_PROXY', '无')}")
    
    try:
        # 使用客户端API初始化
        genai.configure(api_key=api_key)  # 配置API密钥
        
        # 测试简单文本生成
        print("\n测试1: 简单文本生成")
        model = genai.GenerativeModel(model_name.split(':')[0] if ':' in model_name else model_name)
        response = model.generate_content("你好，请介绍一下自己")
        
        if hasattr(response, 'text'):
            print("响应成功!")
            print(f"响应内容: {response.text[:100]}...")
        else:
            print("响应格式异常:")
            print(response)
        
        # 测试图像分析功能
        print("\n测试2: 图像分析")
        # 查找一个图像文件进行测试
        test_images = [
            "static/icon/logo.png",
            "static/img/test.jpg",
            "frames/test.jpg"
        ]
        
        test_image_path = None
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image_path = img_path
                break
        
        if not test_image_path:
            # 如果没有找到测试图像，使用当前目录下的任何jpg/png文件
            for file in os.listdir("."):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image_path = file
                    break
            
            # 如果还是没找到图像，尝试使用视频文件中的一帧
            if not test_image_path and os.path.exists("办公室.mp4"):
                import cv2
                print("未找到图像文件，尝试从视频文件提取帧...")
                cap = cv2.VideoCapture("办公室.mp4")
                ret, frame = cap.read()
                if ret:
                    test_image_path = "test_frame.jpg"
                    cv2.imwrite(test_image_path, frame)
                    print(f"已从视频创建测试图像: {test_image_path}")
                cap.release()
        
        if test_image_path:
            print(f"使用测试图像: {test_image_path}")
            # 读取图像文件
            with open(test_image_path, "rb") as img_file:
                image_data = img_file.read()
            
            # 生成带图像的内容
            image_parts = [
                {
                    "mime_type": "image/jpeg" if test_image_path.lower().endswith(('.jpg', '.jpeg')) else "image/png",
                    "data": image_data
                }
            ]
            
            # 设置生成参数
            generation_config = {
                "temperature": gemini_config.get("temperature", 0.4),
                "max_output_tokens": gemini_config.get("max_output_tokens", 2048)
            }
            
            # 调用图像分析
            model = genai.GenerativeModel(model_name.split(':')[0] if ':' in model_name else model_name)
            response = model.generate_content(
                ["描述这张图片的内容", image_parts[0]],
                generation_config=generation_config
            )
            
            if hasattr(response, 'text'):
                print("图像分析成功!")
                print(f"响应内容: {response.text[:150]}...")
            else:
                print("图像分析响应格式异常:")
                print(response)
        else:
            print("未找到测试图像，跳过图像分析测试")
        
        print("\n✅ Gemini API测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ Gemini API测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Google Gemini API 测试工具")
    print("=" * 50)
    
    # 运行测试
    asyncio.run(test_gemini_integration()) 
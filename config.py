"""
应用配置文件
包含所有配置项，便于集中管理和修改
"""

# config.py
# 重要：请在部署环境中设置以下环境变量以保证API密钥安全：
#   - GEMINI_API_KEY
#   - MINIMAX_API_KEY
#   - DASHSCOPE_API_KEY
#   - MOONSHOT_API_KEY
#   - ARK_API_KEY
#   - YUANJING_API_KEY
#   ...（根据实际模型需求添加）

import os
from pathlib import Path

# 应用基础配置
APP_CONFIG = {
    "debug": os.environ.get("DEBUG", "False").lower() == "true",
    "log_file": os.environ.get("LOG_FILE", "app.log"),
    "upload_dir": os.environ.get("UPLOAD_DIR", "uploads"),
    "frames_dir": os.environ.get("FRAMES_DIR", "frames"),
    "templates_dir": os.environ.get("TEMPLATES_DIR", "templates"),
    "static_dir": os.environ.get("STATIC_DIR", "static"),
    "max_upload_size": int(os.environ.get("MAX_UPLOAD_SIZE", 100 * 1024 * 1024)),  # 默认100MB
    "allowed_video_types": ["video/mp4", "video/quicktime", "video/x-msvideo", "application/octet-stream"],
    "frame_quality": 85,  # JPEG质量
    "max_frame_size": (800, 800),  # 最大帧尺寸
}

# 视频处理配置
VIDEO_CONFIG = {
    "max_frames_per_second": 0.5,  # 每秒处理的最大帧数 (0.5 = 每2秒一帧)
    "preprocess": {
        "contrast_limit": 2.0,
        "tile_grid_size": (8, 8),
        "quality": APP_CONFIG["frame_quality"]
    }
}

# WebSocket配置
WEBSOCKET_CONFIG = {
    "ping_interval": 20,  # 心跳间隔（秒）
    "ping_timeout": 20,   # 心跳超时（秒）
    "close_timeout": 20,  # 关闭超时（秒）
    "max_message_size": 10 * 1024 * 1024  # 最大消息大小（10MB）
}

# 确保所有目录存在
for directory in [APP_CONFIG["upload_dir"], APP_CONFIG["frames_dir"], 
                 APP_CONFIG["templates_dir"], APP_CONFIG["static_dir"]]:
    Path(directory).mkdir(exist_ok=True)

# Gemini 配置
GEMINI_CONFIG = {
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "model": "gemini-2.0-flash",  # 更新为您测试示例中使用的模型
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40,
    "proxy": {
        "http": "http://127.0.0.1:10809",
        "https": "http://127.0.0.1:10809",
        "socks": "socks5://127.0.0.1:10808"
    }  # 更新为完整的代理配置格式
}
# MiniMax 配置
MINIMAX_CONFIG = {
    "api_key": os.environ.get("MINIMAX_API_KEY", ""),
    "group_id": "1910317135855362118",
    "model": "MiniMax-Text-01",
    "base_url": "https://api.minimax.chat/v1",
    "timeout": 30,
    "retry_count": 3,
    "retry_delay": 5
}

# Qwen 配置
QWEN_CONFIG = {
    "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
    "base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    "model": "qwen-vl-max-latest",  
    "timeout": 60,  # 增加超时时间到60秒
    "temperature": 0.7,
    "max_tokens": 2048
}

# Kimi (Moonshot) 配置
KIMI_CONFIG = {
    "api_key": os.environ.get("MOONSHOT_API_KEY", ""),
    "base_url": "https://api.moonshot.cn/v1",
    "model": "moonshot-v1-8k-vision-preview",
    "temperature": 0.1,
    "max_tokens": 1024
}

# Douban (Volcengine) 配置
DOUBAN_CONFIG = {
    "api_key": os.environ.get("ARK_API_KEY", ""),
    "model": "doubao-1-5-vision-pro-32k-250115",  # 更新为最新的豆包视觉模型ID
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",  # 更新为官方文档中的基础URL
    "temperature": 0.7,
    "max_tokens": 1024
}

# 元景大模型配置
LLAMA_SCOUT_CONFIG = {
    "api_key": os.environ.get("YUANJING_API_KEY", ""),
    "model": "Llama-4-Scout-17B-16E-Instruct",
    "base_url": "https://api.yuanjing.com/v1",  # 使用元景的API基础URL
    "temperature": 0.1,
    "max_tokens": 1024
}

# 模型配置
MODELS = {
    "kimi": {
        "name": KIMI_CONFIG["model"],
        "api_key": KIMI_CONFIG["api_key"],
        "base_url": KIMI_CONFIG["base_url"],
        "temperature": KIMI_CONFIG["temperature"],
        "max_tokens": KIMI_CONFIG["max_tokens"]
    },
    "gemini": {
        "name": GEMINI_CONFIG["model"],
        "api_key": GEMINI_CONFIG["api_key"],
        "temperature": GEMINI_CONFIG["temperature"],
        "max_output_tokens": GEMINI_CONFIG["max_output_tokens"],
        "top_p": GEMINI_CONFIG["top_p"],
        "top_k": GEMINI_CONFIG["top_k"]
    },
    "minimax": {
        "name": MINIMAX_CONFIG["model"],
        "api_key": MINIMAX_CONFIG["api_key"],
        "group_id": MINIMAX_CONFIG["group_id"],
        "base_url": MINIMAX_CONFIG["base_url"],
        "timeout": MINIMAX_CONFIG["timeout"],
        "retry_count": MINIMAX_CONFIG["retry_count"],
        "retry_delay": MINIMAX_CONFIG["retry_delay"]
    },
    "qwen": {
        "name": QWEN_CONFIG["model"],
        "api_key": QWEN_CONFIG["api_key"],
        "base_url": QWEN_CONFIG["base_url"],
        "temperature": QWEN_CONFIG["temperature"],
        "max_tokens": QWEN_CONFIG["max_tokens"]
    },
    "douban": {
        "name": DOUBAN_CONFIG["model"],
        "api_key": DOUBAN_CONFIG["api_key"],
        "base_url": DOUBAN_CONFIG["base_url"],
        "temperature": DOUBAN_CONFIG["temperature"],
        "max_tokens": DOUBAN_CONFIG["max_tokens"]
    },
    "llama_scout": {
        "name": LLAMA_SCOUT_CONFIG["model"],
        "api_key": LLAMA_SCOUT_CONFIG["api_key"],
        "base_url": LLAMA_SCOUT_CONFIG["base_url"],
        "temperature": LLAMA_SCOUT_CONFIG["temperature"],
        "max_tokens": LLAMA_SCOUT_CONFIG["max_tokens"]
    }
}

# 默认模型
DEFAULT_MODEL = "kimi"
"""
应用配置文件
包含所有配置项，便于集中管理和修改
"""

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

# Ollama 服务配置
OLLAMA_CONFIG = {
    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://123.157.129.172:3336"),
    "enable_ollama": os.environ.get("ENABLE_OLLAMA", "False").lower() == "true",
    "timeout": int(os.environ.get("OLLAMA_TIMEOUT", 30)),
    "retry_count": int(os.environ.get("OLLAMA_RETRY_COUNT", 2)),
    "retry_delay": int(os.environ.get("OLLAMA_RETRY_DELAY", 3)),
    "health_check_timeout": int(os.environ.get("OLLAMA_HEALTH_CHECK_TIMEOUT", 10)),
    "allow_start_without_ollama": os.environ.get("ALLOW_START_WITHOUT_OLLAMA", "True").lower() == "true"
}

# Ollama API 配置
OLLAMA_API_CONFIG = {
    "base_url": OLLAMA_CONFIG["base_url"],
    "endpoints": {
        "tags": "/api/tags",      # 获取可用模型列表
        "chat": "/api/chat",      # 对话接口
        "generate": "/api/generate", # 生成接口
        "version": "/api/version"  # 版本信息接口
    },
    "timeouts": {
        "connect": 5.0,  # 连接超时
        "read": 30.0,    # 读取超时
        "write": 30.0    # 写入超时
    },
    "models": {
        "vision": "llama3.2-vision:latest",  # 视觉分析模型
        "chat": "llama2"                     # 对话模型
    },
    "retry": {
        "max_attempts": OLLAMA_CONFIG["retry_count"],
        "delay": OLLAMA_CONFIG["retry_delay"]
    }
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
    "api_key": "AIzaSyATUsT0Rp0SUW2qy-s5pUo8sOGHZlsSrRs",  # 更新的API密钥
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
    "api_key": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJZankiLCJVc2VyTmFtZSI6IllqeSIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTEwMzE3MTM1ODYzNzUwNzI2IiwiUGhvbmUiOiIxODY1NzYxODQzMyIsIkdyb3VwSUQiOiIxOTEwMzE3MTM1ODU1MzYyMTE4IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDQtMTAgMjM6NTQ6MDMiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.VVQiLz2t_bN3TG9ra54f4DzPQQQ2NIgtQA_HmX64Yy0JKWWPLQaDQDpYuKgMCmHIi61lLtgsZWfAzCObpfpfMz7_0kLaCAP18Kg4MePOmabFacHukKV3aeBdES7-WtZGgVQEwl7MElM3PCEhd5LBuqbiJNNURA10sFGgk6mrlQ6h_6CigHD46Zmnf1urlPFO-kJRan8vOvSRPTOHOtcT6SU5QNhivggkUU0Lh1aRN9U3Xzuby1Jxb3IEPqyNTUQzLDHIEQuOfM_kNAnC3XjjyeqM4GFSa_UyofjJg67mVhKoiOaERLLqOx8zDK61k2B1x0aVM1gqkwFXFoa-zIIDHA",
    "group_id": "1910317135855362118",
    "model": "MiniMax-Text-01",
    "base_url": "https://api.minimax.chat/v1",
    "timeout": 30,
    "retry_count": 3,
    "retry_delay": 5
}

# Qwen 配置
QWEN_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY", "sk-6b8b715aaea649f986375e79378787f1"),  # 添加之前设置的API密钥作为默认值
    "base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    "model": "qwen-vl-max-latest",  
    "timeout": 60,  # 增加超时时间到60秒
    "temperature": 0.7,
    "max_tokens": 2048
}

# 模型配置
MODELS = {
    "llama": {
        "name": OLLAMA_API_CONFIG["models"]["vision"],  # 使用配置中定义的模型名称
        "base_url": OLLAMA_API_CONFIG["base_url"],     # 使用配置中定义的服务器地址
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
    }
}

# 默认模型
DEFAULT_MODEL = "llama"
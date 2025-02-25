"""
Ollama 服务配置文件
包含服务器连接、超时、重试等相关配置
"""

# 基础配置
OLLAMA_CONFIG = {
    "base_url": "http://123.157.129.172:3336",  # Ollama 服务器地址
    "timeout": 30,  # 请求超时时间（秒）
    "retry_count": 3,  # 重试次数
    "retry_delay": 5,  # 重试间隔（秒）
    "health_check_timeout": 10,  # 健康检查超时时间（秒）
    "allow_start_without_ollama": True  # 是否允许在 Ollama 服务不可用时启动应用
} 

# API 相关配置
OLLAMA_API_CONFIG = {
    "base_url": "http://123.157.129.172:3336",
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
        "max_attempts": 3,  # 最大重试次数
        "delay": 5         # 重试延迟（秒）
    }
} 

# Gemini 配置
GEMINI_CONFIG = {
    "api_key": "AIzaSyDWzg0EHMQZJaABwkQ7RzmxzWtosL01uVk",
    "model": "gemini-2.0-flash-exp",  # 使用新的模型版本
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40,
    "proxy": "http://127.0.0.1:10809"  # 添加代理配置
}

# MiniMax 配置
MINIMAX_CONFIG = {
    "api_key": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJrZW5peCIsIlVzZXJOYW1lIjoia2VuaXgiLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTg3OTE3NDIwMDIzNzc2ODg4MCIsIlBob25lIjoiMTU2NTc1NzYxMDYiLCJHcm91cElEIjoiMTg3OTE3NDIwMDIyOTM4MDI3MiIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI1LTAyLTIwIDE2OjMyOjA1IiwiVG9rZW5UeXBlIjoxLCJpc3MiOiJtaW5pbWF4In0.H77D2A2PDcDydFO6XitXN_27GImSdMp7jqNF8fSN9tWKfjGE1t2lmd8NJw3Z09aizEujfnbOoR_zhuLJEes-_XbLXAHereyLtv1tLMQCxOW6gp0103fLF40R7eSB_knZzZeKWDGvTkicF9eNPWADMPkzbVpKLYbcvuM0wmyjtZCYbs0vGe6C-b0us4eiKb5ZnIUcCYZ6a6hR24LZlNKwlU55oIlLwKo8U3auuYvLoL81P06T3oY2V4QYub8nVx4EG27rGZExfLi5nYXoXvSjC6wLgERx_U6X9Ef_wkOObauv5JT89M6BRxppzfBBYG1szCfjb3Q62eb2Q9M4XRS1Ug",  # 需要替换为实际的 API Key
    "group_id": "1879174200229380272",  # 19位数字的 GroupID
    "model": "abab5.5-chat",
    "base_url": "https://api.minimax.chat/v1",  # 确保这是正确的 API 基础URL
    "timeout": 30,
    "retry_count": 3,
    "retry_delay": 5
}

# Qwen 配置
QWEN_CONFIG = {
    "api_key": "195647d0-b531-42ef-a467-fa036b35ad7f",  # ModelScope Token
    "base_url": "https://api-inference.modelscope.cn/v1/",
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "timeout": 30,
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
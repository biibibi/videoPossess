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
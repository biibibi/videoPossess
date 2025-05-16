# -*- coding: utf-8 -*-
# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
import time
from pathlib import Path
import asyncio
import json
import httpx
from ollama import Client
from contextlib import asynccontextmanager
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import os
from config import OLLAMA_CONFIG, OLLAMA_API_CONFIG, APP_CONFIG, MODELS, DEFAULT_MODEL, GEMINI_CONFIG
import functools
import base64
from openai import OpenAI
from utils.prompt_generator import PromptGenerator
import aiohttp
import datetime
import traceback
import re
import cv2
import numpy as np
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Callable, TypeVar, Awaitable, Optional, Union
import inspect

# 函数返回类型定义
T = TypeVar('T')

# 错误处理装饰器
def api_error_handler(status_code: int = 500):
    """
    API错误处理装饰器，统一处理异常并返回标准化的错误响应
    
    Args:
        status_code: 发生错误时返回的HTTP状态码
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # 获取函数名称用于日志
                func_name = func.__name__
                logger.error(f"{func_name} 错误: {str(e)}")
                traceback_info = traceback.format_exc()
                logger.debug(f"{func_name} 详细错误: {traceback_info}")
                
                # 构建标准错误响应
                return JSONResponse(
                    status_code=status_code,
                    content={
                        "status": "error",
                        "message": str(e),
                        "timestamp": time.time()
                    }
                )
        return wrapper
    return decorator

# 统一API响应生成器
def create_api_response(
    status: str = "success", 
    message: str = None, 
    data: Any = None, 
    status_code: int = 200,
    **extra_fields
) -> JSONResponse:
    """
    创建统一格式的API响应
    
    Args:
        status: 响应状态，'success' 或 'error'
        message: 响应消息
        data: 响应数据
        status_code: HTTP状态码
        extra_fields: 其他需要包含在响应中的字段
        
    Returns:
        格式化的JSONResponse
    """
    response_content = {
        "status": status,
        "timestamp": time.time()
    }
    
    if message:
        response_content["message"] = message
        
    if data is not None:
        response_content["data"] = data
        
    # 添加额外字段
    response_content.update(extra_fields)
    
    return JSONResponse(
        status_code=status_code,
        content=response_content
    )

# 兼容性函数，用于替代 asyncio.to_thread
async def run_in_thread(func, *args, **kwargs):
    """在单独的线程中运行函数"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

# 设置日志
import os
from pathlib import Path

# 确保日志目录存在
log_file_path = APP_CONFIG.get("log_file", "logs/app.log")
log_dir = os.path.dirname(log_file_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG if APP_CONFIG.get("debug", False) else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path)
    ]
)

logger = logging.getLogger(__name__)

# 从配置文件获取设置
OLLAMA_BASE_URL = OLLAMA_CONFIG.get("base_url", "http://123.157.129.172:3336")
ENABLE_OLLAMA = OLLAMA_CONFIG.get("enable_ollama", True)

# 应用配置
TEMPLATES_DIR = Path(APP_CONFIG.get("templates_dir", "templates"))
STATIC_DIR = Path(APP_CONFIG.get("static_dir", "static"))

# 创建必要的目录
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


# 初始化Ollama客户端
ollama_client = None

# 创建Ollama客户端
@retry(stop=stop_after_attempt(OLLAMA_CONFIG.get("retry_count", 2)), 
       wait=wait_exponential(multiplier=1, min=4, max=10))
async def create_ollama_client():
    """
    创建Ollama客户端
    
    使用重试机制确保客户端创建成功
    
    Returns:
        Ollama客户端对象或None（如果创建失败）
    """
    global ollama_client
    if not ENABLE_OLLAMA:
        logger.warning("Ollama服务已禁用，跳过客户端创建")
        return None
            
    try:
        # 创建客户端
        client = Client(host=OLLAMA_BASE_URL)
        
        # 测试连接
        try:
            # 获取可用模型列表
            models = await run_in_thread(client.list)
            
            # 检查所需模型是否可用
            required_model = OLLAMA_API_CONFIG['models']['vision']
            model_names = [m.get('name', '') for m in models.get('models', [])]
            
            if required_model not in model_names:
                logger.warning(f"所需模型 {required_model} 不可用，可用模型: {model_names}")
                return None
                
            logger.info(f"Ollama客户端创建成功，连接到 {OLLAMA_BASE_URL}")
            ollama_client = client
            return client
        except Exception as test_error:
            logger.error(f"测试Ollama连接失败: {str(test_error)}")
            return None
    except Exception as e:
        logger.error(f"创建Ollama客户端失败: {str(e)}")
        return None

async def check_ollama_connection(force_reconnect=False):
    """
    统一的Ollama连接检查和客户端初始化函数
    
    检查Ollama服务连接状态，并在需要时创建或重新创建客户端
    
    Args:
        force_reconnect: 是否强制重新连接，即使客户端已存在
        
    Returns:
        (bool, str): 连接状态和消息元组
    """
    global ollama_client
    
    # 检查是否启用Ollama
    if not ENABLE_OLLAMA:
        return False, "Ollama服务已禁用"
    
    # 强制重连或客户端不存在时，创建新客户端
    if force_reconnect or ollama_client is None:
        logger.info("初始化Ollama客户端" if ollama_client is None else "重新初始化Ollama客户端")
        ollama_client = await create_ollama_client()
        
        if ollama_client is None:
            return False, "无法创建Ollama客户端连接"
    
    # 测试客户端是否可用
    try:
        # 简单的模型列表检查
        models = await run_in_thread(ollama_client.list)
        
        # 检查所需模型是否可用
        required_model = OLLAMA_API_CONFIG['models']['vision']
        model_names = [m.get('name', '') for m in models.get('models', [])]
        
        if required_model not in model_names:
            return False, f"所需模型 {required_model} 不可用"
            
        return True, "Ollama连接正常"
    except Exception as e:
        logger.error(f"Ollama连接测试失败: {str(e)}")
        return False, f"Ollama连接测试失败: {str(e)}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    在应用启动时初始化资源，在应用关闭时清理资源
    """
    # 启动时
    logger.info("应用启动，初始化资源...")
    
    # 初始化Ollama客户端
    if ENABLE_OLLAMA:
        status, message = await check_ollama_connection()
        if not status:
            logger.warning(f"Ollama初始化失败: {message}")
    
    # 确保没有活动的RTSP连接
    await disconnect_rtsp()
    
    yield
    
    # 关闭时
    logger.info("应用关闭，清理资源...")
    
    # 清理Ollama客户端
    global ollama_client
    if ollama_client is not None:
        ollama_client = None
    
    # 断开所有活动的RTSP连接
    await disconnect_rtsp()

# 设置应用生命周期
app = FastAPI(lifespan=lifespan)

# 挂载静态文件 (移到这里)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 设置模板 (移到这里)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 主页路由
@app.get("/")
async def home(request: Request):
    """
    主页路由
    
    渲染主页模板
    """
    return templates.TemplateResponse("index.html", {"request": request})

# 测试Ollama连接
async def test_ollama_connection():
    """
    测试Ollama服务连接
    
    检查Ollama服务是否可用，并初始化客户端
    """
    try:
        # 创建客户端
        client = await create_ollama_client()
        
        if client is None:
            logger.error("无法创建Ollama客户端")
            return False
            
        # 测试连接
        try:
            # 获取可用模型列表
            models = await run_in_thread(client.list)
            
            # 检查所需模型是否可用
            required_model = OLLAMA_API_CONFIG['models']['vision']
            model_names = [m.get('name', '') for m in models.get('models', [])]
            
            if required_model not in model_names:
                logger.warning(f"所需模型 {required_model} 不可用，可用模型: {model_names}")
                return False
                
            logger.info(f"Ollama服务连接成功，可用模型: {model_names}")
            return True
        except Exception as test_error:
            logger.error(f"测试Ollama连接失败: {str(test_error)}")
            return False
    except Exception as e:
        logger.error(f"测试Ollama连接失败: {str(e)}")
        return False

# 检查客户端状态
async def check_client_status():
    """
    检查Ollama客户端状态
    
    如果客户端未初始化或连接失败，尝试重新初始化
    """
    global ollama_client
    
    if ollama_client is None:
        logger.warning("Ollama客户端未初始化，尝试重新初始化")
        ollama_client = await create_ollama_client()
        
        if ollama_client is None:
            logger.error("重新初始化客户端失败")
            return False
    return True

@app.get("/health")
@api_error_handler(status_code=503)
async def health_check():
    """
    健康检查端点
    
    检查Ollama服务连接状态和模型可用性
    """
    # 检查Ollama客户端状态
    status, message = await check_ollama_connection()
    
    if not status:
        return create_api_response(
            status="error",
            message=message,
            status_code=503
        )
    
    # 测试模型功能
    try:
        response = await run_in_thread(
            ollama_client.chat,
            model=OLLAMA_API_CONFIG['models']['vision'],
            messages=[{
                'role': 'user',
                'content': '测试消息'
            }]
        )
        
        # 检查响应格式
        if response and 'message' in response:
            return create_api_response(
                status="success",
                message="服务运行正常",
                ollama_server="已连接",
                model_status="可用",
                model_name=OLLAMA_API_CONFIG['models']['vision']
            )
        else:
            return create_api_response(
                status="error",
                message="模型响应格式无效",
                status_code=503
            )
    except Exception as model_error:
        logger.error(f"模型测试错误: {str(model_error)}")
        return create_api_response(
            status="error",
            message=f"模型测试失败: {str(model_error)}",
            status_code=503
        )

@app.get("/api/test")
async def test_api():
    """
    API测试端点
    
    测试Ollama API连接和可用模型列表
    """
    results = {
        "timestamp": time.time(),
        "server_info": {
            "url": OLLAMA_BASE_URL,
            "connection_status": "unknown",
            "status_code": None
        },
        "available_models": None,
        "error": None
    }
    
    async with httpx.AsyncClient(timeout=OLLAMA_API_CONFIG["timeouts"]["connect"]) as client:
        try:
            # 测试API连接
            tags = await client.get(f"{OLLAMA_BASE_URL}{OLLAMA_API_CONFIG['endpoints']['tags']}")
            results["server_info"]["status_code"] = tags.status_code
            
            if tags.status_code == 200:
                results["server_info"]["connection_status"] = "connected"
                
                # 解析可用模型列表
                try:
                    models_data = tags.json()
                    results["available_models"] = models_data
                    
                    # 检查所需模型是否可用
                    if 'models' in models_data and isinstance(models_data['models'], list):
                        model_names = [m.get('name', '') for m in models_data['models']]
                        required_model = OLLAMA_API_CONFIG['models']['vision']
                        
                        if required_model in model_names:
                            results["required_model"] = {
                                "name": required_model,
                                "available": True
                            }
                        else:
                            results["required_model"] = {
                                "name": required_model,
                                "available": False
                            }
                except json.JSONDecodeError as e:
                    results["error"] = f"解析模型数据失败: {str(e)}"
            else:
                results["server_info"]["connection_status"] = "failed"
                results["error"] = f"API请求失败，状态码: {tags.status_code}"
                
            return {
                "status": "success" if results["error"] is None else "error",
                "results": results
            }
        except Exception as e:
            logger.error(f"API 测试错误: {str(e)}")
            results["error"] = str(e)
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": str(e),
                    "results": results
                }
            )

@app.get("/test-ollama")
async def test_ollama():
    """
    测试Ollama模型的基本对话功能
    """
    try:
        # 检查客户端状态
        if not await check_client_status():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "错误",
                    "message": "Ollama 客户端未初始化",
                    "timestamp": time.time()
                }
            )
            
        # 测试模型对话
        response = await run_in_thread(
            ollama_client.chat,
            model=OLLAMA_API_CONFIG['models']['vision'],
            messages=[{
                'role': 'user',
                'content': '请简单介绍一下你自己'
            }]
        )
        
        # 检查响应格式
        if response and 'message' in response:
            return {
                "status": "成功",
                "model": OLLAMA_API_CONFIG['models']['vision'],
                "response": response,
                "timestamp": time.time()
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "错误",
                    "message": "模型响应格式无效",
                    "timestamp": time.time()
                }
            )
    except Exception as e:
        logger.error(f"测试错误: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "错误",
                "message": str(e),
                "timestamp": time.time()
            }
        )

# 添加图像分析API端点
@app.post("/api/analyze")
@api_error_handler(status_code=500)
async def analyze_image(request: Request):
    """
    图像分析API端点
    
    接收图像和搜索目标，调用相应的模型进行分析
    """
    start_time = time.time()
    
    # 检查请求头中的优先级标记
    priority = request.headers.get("X-Analysis-Priority", "normal")
    frame_timestamp = request.headers.get("X-Frame-Timestamp", "unknown")
    
    # 记录请求时间和优先级
    logger.info(f"收到分析请求: 时间戳={frame_timestamp}, 优先级={priority}, 处理开始时间={start_time}")
    
    # 解析请求数据
    data = await request.json()
    model_name = data.get("model", "")
    image = data.get("image", "")
    targets = data.get("targets", [])
    
    # 验证请求数据
    if not model_name or not image or not targets:
        logger.warning(f"请求参数不完整: model={model_name}, image={'有' if image else '无'}, targets={targets}")
        return create_api_response(
            status="error",
            message="缺少必要的请求参数",
            targets=[{"name": target, "found": False} for target in targets],
            model_name=model_name,
            status_code=400
        )
    
    # 使用统一的模型处理器处理请求
    result = await ModelProcessor.process(model_name, image, targets, start_time)
    
    # 添加处理时间信息
    processing_time = time.time() - start_time
    if isinstance(result, dict):
        result["processing_time"] = processing_time
    # 确保结果始终包含时间戳，保证排序的正确性
    if "timestamp" not in result:
        result["timestamp"] = time.time()
    
    # 确保返回标准JSON格式，使用JSONResponse显式序列化
    return JSONResponse(content=result)

system_prompts = {
    "minimax": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。",
    "gemini": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。",
    "qwen": "你是一个专业的视觉分析助手，请帮助用户分析图像中的内容。",
    "llama": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。"
}

# 添加API调用超时和重试机制
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_api_with_retry(url, headers=None, data=None, method="POST", timeout=30.0):
    """
    使用重试机制调用API
    
    Args:
        url: API地址
        headers: 请求头
        data: 请求数据
        method: 请求方法，默认POST
        timeout: 超时时间，默认30秒
        
    Returns:
        API响应
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        if method.upper() == "POST":
            return await client.post(url, headers=headers, content=data)
        elif method.upper() == "GET":
            return await client.get(url, headers=headers)
        else:
            raise ValueError(f"不支持的HTTP方法: {method}")

# 添加统一响应处理
def format_api_response(content, targets, error=None, model_name=None, found_targets=None):
    """
    统一格式化API响应
    
    Args:
        content: 响应内容
        targets: 搜索目标列表
        error: 错误信息，如果有
        model_name: 模型名称
        found_targets: 已找到的目标列表，如果为None则假设没有找到任何目标
        
    Returns:
        格式化后的响应字典
    """
    # 检查是否为智能分析模式
    is_analysis_mode = False
    user_query = ""
    
    # 检查targets中是否包含智能分析模式标记
    if targets and isinstance(targets, list):
        if len(targets) >= 2 and "mode=intelligent_analysis" in targets:
            is_analysis_mode = True
            # 第一个元素是用户的查询内容
            user_query = targets[0]
            # 处理后的targets仅包含用户查询
            targets = [user_query]
    
    if error:
        if is_analysis_mode:
            return {
                "response": f"处理出错: {error}",
                "description": f"处理出错: {error}",
                "targets": [],
                "model": model_name,
                "status": "error",
                "timestamp": time.time() # 添加时间戳确保排序
            }
        else:
            return {
                "description": f"处理出错: {error}",
                "targets": [{"name": target, "found": False} for target in targets],
                "model": model_name,
                "status": "error",
                "timestamp": time.time() # 添加时间戳确保排序
            }
    
    # 智能分析模式下，我们主要关注响应内容，而不是目标检测
    if is_analysis_mode:
        return {
            "response": content,
            "description": content,
            "targets": [],  # 智能分析模式不需要targets
            "model": model_name,
            "status": "success",
            "timestamp": time.time() # 添加时间戳确保排序
        }
    
    # 以下是原有的目标搜索模式的处理逻辑
    # 如果提供了found_targets列表，则使用它来确定每个目标是否被找到
    if found_targets is not None:
        # 将found_targets转换为小写以便不区分大小写比较
        found_targets_lower = [t.lower() if isinstance(t, str) else t for t in found_targets]
        
        target_results = []
        for target in targets:
            # 检查目标是否在found_targets中（不区分大小写）
            target_lower = target.lower() if isinstance(target, str) else target
            is_found = any(target_lower in ft or ft in target_lower for ft in found_targets_lower)
            target_results.append({"name": target, "found": bool(is_found)})
        
        return {
            "description": content,
            "targets": target_results,
            "model": model_name,
            "status": "success",
            "timestamp": time.time() # 添加时间戳确保排序
        }
    
            # 针对Gemini模型的特殊处理
    if model_name == "gemini":
        # 创建更精确的匹配模式
        content_lower = content.lower() if content else ""
        target_results = []
        
        for target in targets:
            target_lower = target.lower() if isinstance(target, str) else str(target).lower()
            
            # 明确表示物体存在的匹配模式 - 扩展更多匹配模式
            found_patterns = [
                # 中文匹配模式
                f"发现{target_lower}", f"找到{target_lower}", f"检测到{target_lower}",
                f"有{target_lower}", f"存在{target_lower}", f"{target_lower}存在", 
                f"图像中含有{target_lower}", f"图像中有{target_lower}", f"图像包含{target_lower}",
                f"可以看到{target_lower}", f"能够看到{target_lower}", f"能看到{target_lower}",
                # 英文匹配模式 - 添加更多变体以提高匹配率
                f"found {target_lower}", f"detected {target_lower}", f"contains {target_lower}",
                f"has {target_lower}", f"there is {target_lower}", f"there are {target_lower}",
                f"image contains {target_lower}", f"image has {target_lower}",
                f"can see {target_lower}", f"is visible {target_lower}", f"appears {target_lower}",
                f"showing {target_lower}", f"displays {target_lower}", f"present {target_lower}",
                f"i can see {target_lower}", f"we can see {target_lower}", f"there's {target_lower}"
            ]
            
            # 明确表示物体不存在的匹配模式 - 扩展更多匹配模式
            not_found_patterns = [
                # 中文匹配模式
                f"没有{target_lower}", f"未发现{target_lower}", f"未找到{target_lower}", 
                f"没有看到{target_lower}", f"未检测到{target_lower}", f"不存在{target_lower}",
                f"{target_lower}未出现", f"{target_lower}不存在", f"{target_lower}没有",
                f"无法看到{target_lower}", f"看不到{target_lower}", f"没有{target_lower}的踪影",
                # 英文匹配模式 - 添加更多变体以提高匹配率
                f"no {target_lower}", f"not found {target_lower}", f"cannot see {target_lower}",
                f"doesn't contain {target_lower}", f"does not contain {target_lower}",
                f"didn't find {target_lower}", f"could not find {target_lower}",
                f"wasn't able to find {target_lower}", f"is not present {target_lower}",
                f"isn't visible {target_lower}", f"not visible {target_lower}",
                f"not showing {target_lower}", f"absent {target_lower}",
                f"i don't see {target_lower}", f"i cannot find {target_lower}",
                f"couldn't locate {target_lower}", f"not in the image {target_lower}"
            ]
            
            # 提取与目标相关的上下文
            context_window = 50  # 上下文窗口大小
            target_positions = []
            
            # 找到目标在内容中的所有位置
            pos = content_lower.find(target_lower)
            while pos != -1:
                target_positions.append(pos)
                pos = content_lower.find(target_lower, pos + 1)
            
            # 默认为未找到
            is_found = False
            
            # 首先检查是否有明确的"找到"模式
            if any(pattern in content_lower for pattern in found_patterns):
                is_found = True
                logger.debug(f"Gemini找到目标 '{target}' - 匹配明确的找到模式")
            # 如果没有明确的"找到"模式，检查目标词是否在内容中
            elif target_positions:
                # 检查每个目标出现位置的上下文
                has_not_found_context = False
                
                for pos in target_positions:
                    # 提取目标词周围的上下文
                    start = max(0, pos - context_window)
                    end = min(len(content_lower), pos + len(target_lower) + context_window)
                    context = content_lower[start:end]
                    
                    # 检查上下文中是否有"未找到"模式
                    if any(pattern in context for pattern in not_found_patterns):
                        has_not_found_context = True
                        logger.debug(f"Gemini目标 '{target}' 的上下文包含否定模式")
                        break
                
                # 如果周围上下文没有否定模式，则认为找到了目标
                if not has_not_found_context:
                    is_found = True
                    logger.debug(f"Gemini找到目标 '{target}' - 目标词在内容中且无否定上下文")
            
            # 记录最终结果
            logger.info(f"Gemini模型 - 目标 '{target}' 的查找结果: {is_found}")
            target_results.append({"name": target, "found": bool(is_found)})
            
        return {
            "description": content,
            "targets": target_results,
            "model": model_name,
            "status": "success",
            "timestamp": time.time() # 添加时间戳确保排序
        }
    
    # 原有的启发式方法，用于其他模型
    content_lower = content.lower() if content else ""
    target_results = []
    
    for target in targets:
        target_lower = target.lower() if isinstance(target, str) else str(target).lower()
        # 检查内容中是否包含"没有发现"+目标名称的模式
        not_found_patterns = [
            f"没有{target_lower}", f"未发现{target_lower}", f"未找到{target_lower}", 
            f"没有看到{target_lower}", f"未检测到{target_lower}", f"不存在{target_lower}",
            f"{target_lower}未出现", f"{target_lower}不存在", f"{target_lower}没有",
            "no " + target_lower, "not found " + target_lower, "cannot see " + target_lower,
            "doesn't contain " + target_lower, "does not contain " + target_lower
        ]
        
        # 如果有任何一个"未找到"的模式匹配，则认为目标未找到
        is_found = not any(pattern in content_lower for pattern in not_found_patterns)
        
        # 如果内容中明确提到了目标名称，更可能是找到了
        if target_lower in content_lower and is_found:
            # 检查是否在目标名称附近有否定词
            for pattern in not_found_patterns:
                if pattern in content_lower:
                    is_found = False
                    break
        
        target_results.append({"name": target, "found": bool(is_found)})
    
    return {
        "description": content,
        "targets": target_results,
        "model": model_name,
        "status": "success",
        "timestamp": time.time() # 添加时间戳确保排序
    }

# 添加模型处理器类，统一处理不同模型的逻辑
class ModelProcessor:
    """模型处理器类，统一处理不同模型的API调用"""
    
    @staticmethod
    async def _extract_user_message_text(user_message) -> str:
        """
        从用户消息中提取文本内容
        
        Args:
            user_message: 用户消息对象
            
        Returns:
            提取的文本内容
        """
        text_content = ""
        if user_message and isinstance(user_message, dict) and "content" in user_message:
            if isinstance(user_message["content"], list) and len(user_message["content"]) > 0:
                if isinstance(user_message["content"][0], dict) and "text" in user_message["content"][0]:
                    text_content = user_message["content"][0]["text"]
                elif user_message["content"][0] is not None:
                    text_content = str(user_message["content"][0])
            elif isinstance(user_message["content"], str):
                text_content = user_message["content"]
        
        return text_content
    
    @staticmethod
    async def _extract_system_message_text(system_message) -> str:
        """
        从系统消息中提取文本内容
        
        Args:
            system_message: 系统消息对象
            
        Returns:
            提取的文本内容
        """
        if system_message and isinstance(system_message, dict) and "content" in system_message:
            return system_message["content"]
        return "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。"
    
    @staticmethod
    async def _generate_default_query(targets) -> str:
        """
        根据目标列表生成默认查询文本
        
        Args:
            targets: 搜索目标列表
            
        Returns:
            生成的查询文本
        """
        target_text = "、".join(targets) if targets else "内容"
        return f"请分析图像中是否含有以下内容：{target_text}"
    
    @staticmethod
    async def _parse_json_from_text(content: str):
        """
        从文本中提取和解析JSON内容
        
        Args:
            content: 包含可能JSON内容的文本
            
        Returns:
            解析后的JSON对象，或None
        """
        try:
            # 尝试识别和提取JSON内容
            json_match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            json_text = json_match.group(1).strip() if json_match else content
            
            # 清理可能的非JSON文本
            json_text = re.sub(r'^[^{]*', '', json_text)
            json_text = re.sub(r'[^}]*$', '', json_text)
            
            # 尝试解析
            return json.loads(json_text)
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.warning(f"JSON解析失败: {str(e)}")
            return None
    
    @staticmethod
    async def process_minimax(image_data, targets, system_message, user_message, start_time):
        """处理Minimax模型的API调用"""
        try:
            # 获取Minimax配置
            minimax_config = MODELS["minimax"]
            api_key = minimax_config["api_key"]
            
            # 记录API调用开始
            api_start_time = time.time()
            logger.info(f"开始调用minimax模型API，时间: {api_start_time}")
            
            # 提取消息内容
            text_content = await ModelProcessor._extract_user_message_text(user_message)
            if not text_content:
                text_content = await ModelProcessor._generate_default_query(targets)
                logger.warning(f"无法从user_message提取文本内容，使用默认查询: {text_content}")
            
            system_content = await ModelProcessor._extract_system_message_text(system_message)
            
            # 初始化OpenAI客户端，使用Minimax的API
            client = OpenAI(
                api_key=api_key,
                base_url=minimax_config["base_url"],
                timeout=float(minimax_config.get("timeout", 30.0))
            )
            
            # 发送请求
            response = client.chat.completions.create(
                model=minimax_config["name"],
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": [
                        {"type": "text", "text": text_content},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]}
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            # 计算API调用时间
            api_call_time = time.time() - api_start_time
            logger.info(f"模型minimax API调用完成，耗时: {api_call_time:.2f}秒")
            
            # 提取响应内容
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                logger.debug(f"Minimax API 原始返回: {content}")
                
                # 尝试解析JSON返回
                parsed_json = await ModelProcessor._parse_json_from_text(content)
                
                if parsed_json is not None:
                    # 提取关键信息
                    description = parsed_json.get("description", content)
                    parsed_targets = parsed_json.get("targets", [])
                    
                    # 如果成功解析了targets
                    if parsed_targets:
                        return {
                            "description": description,
                            "targets": parsed_targets,
                            "model": "minimax",
                            "status": "success",
                            "timestamp": time.time()  # 添加时间戳确保排序
                        }
                
                # 使用通用响应解析
                return format_api_response(content, targets, model_name="minimax")
            else:
                # 响应没有有效的choices
                error_msg = "API响应缺少有效的choices"
                logger.warning(f"Minimax API 响应无效: {error_msg}")
                content = f"API响应格式无效: {str(response)}"
                
            # 使用标准格式化返回结果
            return format_api_response(content, targets, model_name="minimax")
                
        except Exception as api_error:
            logger.error(f"调用模型minimax API时出错: {str(api_error)}")
            traceback_info = traceback.format_exc()
            logger.debug(f"调用minimax出错详细信息: {traceback_info}")
            return format_api_response(None, targets, error=f"{str(api_error)}", model_name="minimax")
    
    @staticmethod
    async def process_qwen(image_data, targets, system_message, user_message, start_time):
        """处理千问模型的API调用"""
        try:
            # 获取千问配置
            qwen_config = MODELS["qwen"]
            api_key = qwen_config["api_key"]
            
            # 提取消息内容
            text_content = await ModelProcessor._extract_user_message_text(user_message)
            if not text_content:
                text_content = await ModelProcessor._generate_default_query(targets)
                logger.warning(f"无法从user_message提取文本内容，使用默认查询: {text_content}")
            
            # 构建请求数据 - 使用通义千问多模态API格式
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建通义千问格式的请求体
            request_body = {
                "model": "qwen-vl-max-2025-04-02",
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个专业的图像分析助手，请帮我分析以下问题："
                        },
                        {
                            "role": "user",
                            "content": [
                                {"image": f"data:image/jpeg;base64,{image_data}"},
                                {"text": text_content}
                            ]
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message",
                    "max_tokens": 1500
                }
            }
            
            # 记录API调用开始
            api_start_time = time.time()
            logger.info(f"开始调用qwen模型API，时间: {api_start_time}")
            
            # 使用UTF-8明确编码发送请求
            request_json = json.dumps(request_body, ensure_ascii=False)
            
            # 发送请求，使用重试机制
            response = await call_api_with_retry(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                headers=headers,
                data=request_json.encode('utf-8'),
                timeout=30.0
            )
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"调用qwenAPI失败，状态码: {response.status_code}, 响应: {response.text}"
                logger.error(error_msg)
                return format_api_response(None, targets, error=error_msg, model_name="qwen")
            
            # 解析响应
            result_data = response.json()
            
            # 计算API调用时间
            api_call_time = time.time() - api_start_time
            logger.info(f"模型qwen API调用完成，耗时: {api_call_time:.2f}秒")
            
            # 提取响应内容
            content = ""
            raw_text = ""
            
            try:
                # 处理嵌套结构
                choices = result_data.get("output", {}).get("choices", [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message = choices[0].get("message", {})
                    message_content = message.get("content", [])
                    if message_content and isinstance(message_content, list) and len(message_content) > 0:
                        raw_text = message_content[0].get("text", "")
                
                # 使用raw_text，或者原始结果字符串
                content = raw_text or str(result_data)
                
                # 尝试解析JSON内容
                parsed_json = await ModelProcessor._parse_json_from_text(content)
                
                if parsed_json is not None:
                    # 提取描述和目标
                    description = parsed_json.get("description", content)
                    parsed_targets = parsed_json.get("targets", [])
                    
                    # 如果成功解析了targets
                    if parsed_targets:
                        return {
                            "description": description,
                            "targets": parsed_targets,
                            "model": "qwen",
                            "status": "success",
                            "timestamp": time.time()  # 添加时间戳确保排序
                        }
            except Exception as extract_error:
                logger.error(f"提取Qwen响应内容时出错: {str(extract_error)}")
                content = str(result_data)
            
            # 使用通用响应解析
            return format_api_response(content, targets, model_name="qwen")
                
        except Exception as api_error:
            error_msg = f"调用模型qwen API时出错: {str(api_error)}"
            logger.error(error_msg)
            traceback_info = traceback.format_exc()
            logger.debug(f"调用qwen出错详细信息: {traceback_info}")
            return format_api_response(None, targets, error=error_msg, model_name="qwen")
    
    @staticmethod
    async def process_gemini(image_data, targets, system_message, user_message, start_time):
        """处理Google Gemini模型的API调用"""
        try:
            # 获取Gemini配置
            gemini_config = MODELS["gemini"]
            api_key = gemini_config["api_key"]
            model_name = gemini_config.get("name", "gemini-2.0-flash-exp")
            proxy = GEMINI_CONFIG.get("proxy", {})
            
            # 记录API调用开始
            api_start_time = time.time()
            logger.info(f"开始调用Gemini模型API，时间: {api_start_time}")
            
            # 提取消息内容
            text_content = await ModelProcessor._extract_user_message_text(user_message)
            if not text_content:
                text_content = await ModelProcessor._generate_default_query(targets)
                logger.warning(f"无法从user_message提取文本内容，使用默认查询: {text_content}")
            
            system_content = await ModelProcessor._extract_system_message_text(system_message)
            
            # 使用更可靠的线程模式 - 注意这是一个普通函数，不是协程
            def run_gemini_api():
                # 导入google.generativeai库
                import google.generativeai as genai
                import os
                
                # 设置代理环境变量
                original_http_proxy = os.environ.get('HTTP_PROXY')
                original_https_proxy = os.environ.get('HTTPS_PROXY')
                
                try:
                    # 应用代理设置（如果有）
                    if proxy.get("http"):
                        os.environ['HTTP_PROXY'] = proxy.get("http")
                    if proxy.get("https"):
                        os.environ['HTTPS_PROXY'] = proxy.get("https")
                    
                    # 配置API密钥
                    genai.configure(api_key=api_key)
                    
                    # 创建临时图像文件 
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                        temp_path = temp_file.name
                        # 将base64图像数据解码为二进制并写入临时文件
                        image_binary = base64.b64decode(image_data)
                        temp_file.write(image_binary)
                    
                    try:
                        # 读取图像文件
                        with open(temp_path, "rb") as img_file:
                            image_data_bytes = img_file.read()
                        
                        # 构建更结构化的提示，请求输出JSON格式
                        structured_prompt = f"{system_content}\n\n请分析图像中是否包含以下目标：{', '.join(targets)}\n\n{text_content}\n\n重要：请确保回答中明确标明每个目标的found状态为true或false，可以使用JSON格式返回结果。例如: {{\"description\": \"描述...\", \"targets\": [{{\"name\": \"目标名\", \"found\": true}}]}}"
                        
                        # 构建图像部分
                        image_part = {
                            "mime_type": "image/jpeg",
                            "data": image_data_bytes
                        }
                        
                        # 构建生成参数
                        generation_config = {
                            "temperature": gemini_config.get("temperature", 0.4),
                            "top_p": gemini_config.get("top_p", 0.8),
                            "top_k": gemini_config.get("top_k", 40),
                            "max_output_tokens": gemini_config.get("max_output_tokens", 2048)
                        }
                        
                        # 创建模型实例
                        model = genai.GenerativeModel(
                            model_name.split(':')[0] if ':' in model_name else model_name
                        )
                        
                        # 调用模型进行图像分析，使用结构化提示
                        response = model.generate_content(
                            [structured_prompt, image_part],
                            generation_config=generation_config
                        )
                        
                        # 处理响应
                        if hasattr(response, 'text'):
                            return response.text
                        elif hasattr(response, 'parts'):
                            # 处理多部分响应
                            return ''.join(part.text for part in response.parts if hasattr(part, 'text'))
                        else:
                            return str(response)
                    finally:
                        # 清理临时文件
                        try:
                            os.unlink(temp_path)
                        except Exception as e:
                            logger.warning(f"清理Gemini临时文件失败: {e}")
                finally:
                    # 恢复原始代理设置
                    if original_http_proxy:
                        os.environ['HTTP_PROXY'] = original_http_proxy
                    elif 'HTTP_PROXY' in os.environ:
                        del os.environ['HTTP_PROXY']
                        
                    if original_https_proxy:
                        os.environ['HTTPS_PROXY'] = original_https_proxy
                    elif 'HTTPS_PROXY' in os.environ:
                        del os.environ['HTTPS_PROXY']
            
            # 使用run_in_thread函数进行线程调用
            content = await run_in_thread(run_gemini_api)
            
            # 计算API调用时间
            api_call_time = time.time() - api_start_time
            logger.info(f"模型Gemini API调用完成，耗时: {api_call_time:.2f}秒")
            
            # 记录原始响应
            logger.debug(f"Gemini API 原始返回: {content}")
            
            # 尝试解析JSON返回
            parsed_json = await ModelProcessor._parse_json_from_text(content)
            
            if parsed_json is not None:
                # 提取关键信息
                description = parsed_json.get("description", content)
                parsed_targets = parsed_json.get("targets", [])
                
                # 如果成功解析了targets
                if parsed_targets:
                    return {
                        "description": description,
                        "targets": parsed_targets,
                        "model": "gemini",
                        "status": "success",
                        "timestamp": time.time()  # 添加时间戳确保排序
                    }
            
            # 如果无法解析为JSON，或者没有找到targets数组，进行更多处理
            # 使用更精细的解析机制，类似Qwen模型的处理方式
            try:
                # 提取内容部分，移除多余的格式信息
                cleaned_content = content.strip()
                
                # 创建结构化的响应，就像Qwen那样
                structured_response = {
                    "description": cleaned_content,
                    "targets": [],  # 将通过format_api_response填充
                    "model": "gemini",
                    "status": "success",
                    "timestamp": time.time()
                }
                
                # 使用format_api_response处理目标检测
                return format_api_response(cleaned_content, targets, model_name="gemini")
                
            except Exception as parse_error:
                logger.error(f"处理Gemini响应时出错: {str(parse_error)}")
                return format_api_response(content, targets, model_name="gemini")  # 回退到简单处理
                
        except Exception as api_error:
            logger.error(f"调用模型Gemini API时出错: {str(api_error)}")
            traceback_info = traceback.format_exc()
            logger.debug(f"调用Gemini出错详细信息: {traceback_info}")
            return format_api_response(None, targets, error=f"{str(api_error)}", model_name="gemini")
    
    @staticmethod
    async def process_unsupported(model_name, targets, start_time):
        """处理尚未支持的模型"""
        content = f"模型{model_name}的响应处理尚未完成实现。检测到{len(targets)}个搜索目标。"
        return format_api_response(content, targets, model_name=model_name)
    
    @classmethod
    @api_error_handler(status_code=500)
    async def process(cls, model_name, image, targets, start_time):
        """
        根据模型名称选择相应的处理方法
        
        Args:
            model_name: 模型名称
            image: 图像base64数据
            targets: 搜索目标列表
            start_time: 请求开始时间
            
        Returns:
            处理结果
        """
        # 参数验证
        if not model_name:
            logger.warning("没有提供模型名称，使用默认模型")
            model_name = "minimax"  # 默认使用minimax模型
        
        if not image:
            logger.error("没有提供图像数据")
            return format_api_response(None, targets, error="没有提供图像数据", model_name=model_name)
        
        # 处理搜索目标
        if not targets or not isinstance(targets, list):
            logger.warning("无效的搜索目标列表，使用默认目标")
            targets = ["人物", "物体"]
        
        # 处理base64图像数据
        try:
            image_data = image.split(',')[1] if ',' in image else image
        except Exception as e:
            logger.error(f"处理图像数据失败: {str(e)}")
            return format_api_response(None, targets, error="图像数据格式无效", model_name=model_name)
        
        # 使用PromptGenerator生成提示词
        try:
            system_message, user_message = PromptGenerator.generate_messages(targets, model_name)
        except Exception as e:
            logger.error(f"生成提示词失败: {str(e)}")
            # 创建默认消息
            system_message = {"role": "system", "content": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。"}
            target_text = "、".join(targets) if targets else "内容"
            user_message = {
                "role": "user", 
                "content": [{"type": "text", "text": f"请分析图像中是否含有以下内容：{target_text}"}]
            }
        
        # 根据模型选择处理方法
        if model_name == "minimax":
            return await cls.process_minimax(image_data, targets, system_message, user_message, start_time)
        elif model_name == "qwen":
            return await cls.process_qwen(image_data, targets, system_message, user_message, start_time)
        elif model_name == "gemini":
            return await cls.process_gemini(image_data, targets, system_message, user_message, start_time)
        else:
            return await cls.process_unsupported(model_name, targets, start_time)

# 添加全局变量来保存RTSP连接和视频流
rtsp_cap = None
rtsp_url = None
rtsp_connected = False
rtsp_frame = None
rtsp_last_frame_time = 0
output_frame = None  # 当前输出帧
last_screenshot_frame = None  # 最后一个截图帧
last_screenshot_time = 0  # 最后截图时间
lock = asyncio.Lock()  # 用于保护全局帧变量的锁
rtsp_thread = None  # RTSP帧捕获线程
JPEG_QUALITY = 85  # JPEG图像质量
SCREENSHOT_INTERVAL = 1.0  # 截图间隔，秒
FRAME_WIDTH = 640  # 帧宽度，如果需要调整
FRAME_HEIGHT = 360  # 帧高度，如果需要调整
capture_task = None  # 异步捕获任务

# RTSP帧获取通用函数
@app.get("/api/rtsp/frame")
@api_error_handler(status_code=500)
async def get_rtsp_frame(source: str = "current"):
    """
    获取RTSP视频流的当前帧或截图
    
    Args:
        source: 帧来源，"current"表示当前帧，"screenshot"表示截图帧
        
    Returns:
        包含帧base64编码的响应
    """
    global rtsp_connected, output_frame, last_screenshot_frame
    
    async with lock:
        # 根据来源选择使用的帧
        if source == "screenshot":
            frame = last_screenshot_frame
        else:  # 默认使用当前帧
            frame = output_frame
            
        if not rtsp_connected or frame is None:
            return create_api_response(
                status="error",
                message="未连接到RTSP流或未获取到有效帧",
                connected=False,
                frame=None,
                status_code=404
            )
        
        # 将帧编码为base64字符串
        success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not success:
            return create_api_response(
                status="error",
                message="无法编码视频帧",
                connected=True,
                frame=None,
                status_code=500
            )
            
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return create_api_response(
            status="success",
            connected=True,
            frame=f"data:image/jpeg;base64,{frame_base64}"
        )

# 保留原有截图API，但改为调用通用函数
@app.get("/api/rtsp/screenshot/base64")
async def get_rtsp_screenshot_base64():
    """
    获取RTSP视频流的最新截图(Base64编码)
    
    返回Base64编码的图像数据，适用于分析API
    """
    return await get_rtsp_frame(source="screenshot")

# 添加RTSP截图路由
@app.get("/api/rtsp/screenshot")
@api_error_handler(status_code=500)
async def get_rtsp_screenshot():
    """
    获取RTSP视频流的最新截图
    
    返回JPEG格式的图像数据
    """
    global rtsp_connected, last_screenshot_frame
    
    async with lock:
        if not rtsp_connected or last_screenshot_frame is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "截图不可用，RTSP流未连接或尚未获取到有效帧",
                    "connected": False
                }
            )
        
        # 复制帧以避免并发修改问题
        frame_copy = last_screenshot_frame.copy()
        
    # 将帧编码为JPEG图像
    success, buffer = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not success:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "无法编码截图",
                "connected": True
            }
        )
        
    # 返回JPEG图像
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

# RTSP连接路由
@app.post("/api/rtsp/connect")
@api_error_handler(status_code=500)
async def connect_rtsp(request: Request):
    """
    连接到RTSP视频流
    
    接收RTSP URL，使用OpenCV连接到视频流
    """
    global rtsp_cap, rtsp_url, rtsp_connected, rtsp_frame, rtsp_last_frame_time, capture_task
    
    # 解析请求数据
    data = await request.json()
    url = data.get("url", "")
    
    # 验证URL
    if not url or not url.startswith("rtsp://"):
        return create_api_response(
            status="error",
            message="无效的RTSP URL",
            connected=False,
            status_code=400
        )
    
    # 如果已经连接到相同的URL，直接返回成功
    if rtsp_connected and rtsp_url == url and rtsp_cap is not None and rtsp_cap.isOpened():
        return create_api_response(
            status="success",
            message="已经连接到该RTSP流",
            connected=True
        )
    
    # 如果之前有连接，先关闭
    await disconnect_rtsp()
    
    # 重置连接状态
    rtsp_connected = False
    rtsp_frame = None
    rtsp_last_frame_time = 0
    
    # 连接到新的RTSP URL
    logger.info(f"尝试连接到RTSP流: {url}")
    rtsp_url = url
    
    # 启动异步捕获任务
    capture_task = asyncio.create_task(capture_frames())
    
    # 等待一段时间以确保能够成功连接
    await asyncio.sleep(2)
    
    # 检查连接状态
    if not rtsp_connected:
        logger.error(f"无法连接到RTSP流: {url}")
        if capture_task:
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass
            capture_task = None
            
    return create_api_response(
        status="error",
        message="无法连接到RTSP流，请检查URL是否正确",
        connected=False,
        status_code=500
    )

# RTSP断开连接路由
@app.post("/api/rtsp/disconnect")
@api_error_handler(status_code=500)
async def disconnect_rtsp():
    """
    断开RTSP视频流连接
    """
    global rtsp_cap, rtsp_url, rtsp_connected, rtsp_frame, capture_task
    
    # 取消捕获任务
    if capture_task:
        capture_task.cancel()
        try:
            await capture_task
        except asyncio.CancelledError:
            pass
        capture_task = None
    
    # 关闭视频捕获对象
    if rtsp_cap is not None:
        # 简化为直接调用，避免try嵌套
        await run_in_thread(lambda: rtsp_cap.release())
        logger.info("视频捕获对象已释放")
        rtsp_cap = None
    
    # 添加短暂延迟，尝试允许底层资源完全释放
    await asyncio.sleep(0.1)
    
    # 重置状态变量
    rtsp_url = None
    rtsp_connected = False
    rtsp_frame = None
    
    return create_api_response(
        status="success",
        message="成功断开RTSP连接",
        connected=False
    )

# RTSP状态检查路由
@app.get("/api/rtsp/status")
@api_error_handler(status_code=500)
async def check_rtsp_status():
    """
    检查RTSP连接状态
    """
    global rtsp_cap, rtsp_url, rtsp_connected
    
    if rtsp_cap is not None and rtsp_connected:
        is_opened = await run_in_thread(lambda: rtsp_cap.isOpened())
        if not is_opened:
            rtsp_connected = False
    
    return create_api_response(
        status="success",
        connected=rtsp_connected,
        url=rtsp_url if rtsp_connected else None,
        last_frame_time=rtsp_last_frame_time if rtsp_connected else 0
    )

# 新增MJPEG流端点
@app.get("/api/rtsp/stream")
async def video_feed():
    """
    提供MJPEG视频流
    """
    if not rtsp_connected:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": "未连接到RTSP流",
                "connected": False
            }
        )
    
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# 初始化视频捕获对象
async def initialize_video_capture():
    """
    初始化或重新初始化视频捕获对象
    """
    global rtsp_cap, rtsp_url, rtsp_connected
    
    logger.info(f"尝试连接到RTSP流: {rtsp_url}")
    
    if rtsp_cap is not None:
        await run_in_thread(lambda: rtsp_cap.release())
    
    # 使用OpenCV连接RTSP流
    rtsp_cap = await run_in_thread(lambda: cv2.VideoCapture(rtsp_url))
    
    # 检查连接是否成功
    is_opened = await run_in_thread(lambda: rtsp_cap.isOpened())
    if not is_opened:
        logger.error(f"无法连接到RTSP流: {rtsp_url}")
        rtsp_connected = False
        return False
    
    # 可选：设置缓冲区大小以减少延迟
    await run_in_thread(lambda: rtsp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 3))
    
    rtsp_connected = True
    logger.info(f"成功连接到RTSP流: {rtsp_url}")
    return True

# 持续捕获帧的异步任务
async def capture_frames():
    """
    持续从RTSP流中捕获帧的异步任务
    """
    global rtsp_cap, rtsp_connected, output_frame, last_screenshot_frame, last_screenshot_time
    
    logger.info("启动RTSP帧捕获任务")
    
    # 初始化视频捕获
    success = await initialize_video_capture()
    if not success:
        logger.error("初始化视频捕获失败")
        rtsp_connected = False
        return
    
    # 主捕获循环
    while rtsp_connected:
        try:
            # 检查连接状态
            is_opened = await run_in_thread(lambda: rtsp_cap.isOpened() if rtsp_cap else False)
            if not is_opened:
                logger.warning("RTSP连接已断开，尝试重新连接")
                rtsp_connected = False
                success = await initialize_video_capture()
                if not success:
                    # 如果重连失败，等待一段时间后再次尝试
                    await asyncio.sleep(5)
                    continue
            
            # 读取一帧
            ret, frame = await run_in_thread(lambda: rtsp_cap.read() if rtsp_cap else (False, None))
            
            if not ret or frame is None:
                # 读取失败，可能是网络问题，等待一会再试
                logger.warning("RTSP帧读取失败")
                await asyncio.sleep(0.5)
                continue
            
            # 可选：调整帧大小
            if FRAME_WIDTH and FRAME_HEIGHT:
                frame = await run_in_thread(lambda: cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)))
            
            # 更新全局帧
            current_time = time.time()
            async with lock:
                output_frame = frame.copy()
                # 更新截图帧（如果间隔已过）
                if current_time - last_screenshot_time >= SCREENSHOT_INTERVAL:
                    last_screenshot_frame = frame.copy()
                    last_screenshot_time = current_time
            
            # 控制循环速率，避免过度占用CPU
            await asyncio.sleep(0.01)
            
        except asyncio.CancelledError:
            # 任务被取消
            logger.info("RTSP帧捕获任务被取消")
            raise
        except Exception as e:
            # 其他错误
            logger.error(f"RTSP帧捕获出错: {str(e)}")
            traceback.print_exc()
            await asyncio.sleep(1)  # 出错后等待较长时间再试
    
    # 清理工作
    if rtsp_cap:
        await run_in_thread(lambda: rtsp_cap.release())
    
    logger.info("RTSP帧捕获任务结束")

# 生成视频流
async def generate_video_stream():
    """
    生成用于web页面的MJPEG流
    """
    global output_frame, rtsp_connected
    
    while rtsp_connected:
        try:
            frame_copy = None
            async with lock:
                if output_frame is None:
                    # 如果没有可用帧，等待
                    await asyncio.sleep(0.1)
                    continue
                frame_copy = output_frame.copy()
            
            # 编码帧为JPEG
            success, buffer = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not success:
                await asyncio.sleep(0.1)
                continue
            
            # 生成帧数据
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # 控制帧率
            await asyncio.sleep(1/30)  # 约30FPS
            
        except Exception as e:
            logger.error(f"生成视频流时出错: {str(e)}")
            await asyncio.sleep(0.5)
            
    # 当连接断开时，发送一个黑色帧或消息帧
    try:
        # 创建一个黑色帧
        black_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        cv2.putText(black_frame, "RTSP Stream Disconnected", (50, FRAME_HEIGHT//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        success, buffer = cv2.imencode('.jpg', black_frame)
        if success:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        logger.error(f"生成最终帧时出错: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import sys
    import platform
    
    # 打印启动信息
    print("=" * 50)
    print(f"启动 Vi-Qwen-gemini-llama 应用")
    print(f"Python 版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"工作目录: {os.getcwd()}")
    print(f"Ollama 服务: {'启用' if ENABLE_OLLAMA else '禁用'}")
    print(f"Ollama 服务器: {OLLAMA_BASE_URL}")
    print(f"模板目录: {TEMPLATES_DIR}")
    print(f"静态文件目录: {STATIC_DIR}")
    print("=" * 50)
    
    # 检查必要的目录
    for directory in [TEMPLATES_DIR, STATIC_DIR]:
        if not directory.exists():
            print(f"警告: 目录不存在，将创建: {directory}")
            directory.mkdir(exist_ok=True)
    
    # 检查Ollama服务
    if ENABLE_OLLAMA:
        print("正在检查Ollama服务...")
        import asyncio
        try:
            # 创建一个事件循环来运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 运行测试连接函数
            result = loop.run_until_complete(test_ollama_connection())
            
            if result:
                print("Ollama服务连接成功")
            else:
                print("警告: Ollama服务连接失败，应用可能无法正常工作")
                
            # 关闭事件循环
            loop.close()
        except Exception as e:
            print(f"错误: 检查Ollama服务时出错: {str(e)}")
    
    # 启动应用
    print("正在启动应用...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info" if not APP_CONFIG.get("debug", False) else "debug"
    )
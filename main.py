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
from contextlib import asynccontextmanager
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import os
from config import APP_CONFIG, MODELS, DEFAULT_MODEL, GEMINI_CONFIG
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

# 导入特定模型的依赖
try:
    import volcenginesdkarkruntime  # 火山引擎豆包SDK
    ARK_IMPORTED = True
    logging.info("volcengine豆包SDK导入成功")
except ImportError:
    ARK_IMPORTED = False
    logging.warning("volcenginesdkarkruntime未安装，豆包模型将无法使用，请执行 pip install volcengine-python-sdk")

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
# 应用配置
TEMPLATES_DIR = Path(APP_CONFIG.get("templates_dir", "templates"))
STATIC_DIR = Path(APP_CONFIG.get("static_dir", "static"))

# 创建必要的目录
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

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

# RTSP断开连接函数 - 定义在lifespan前，因为lifespan依赖它
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
    
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    在应用启动时初始化资源，在应用关闭时清理资源
    """
    # 启动时
    logger.info("应用启动，初始化资源...")
    
    # 确保没有活动的RTSP连接
    await disconnect_rtsp()
    
    yield
    
    # 关闭时
    logger.info("应用关闭，清理资源...")
    
    # 断开所有活动的RTSP连接
    await disconnect_rtsp()

# 设置应用生命周期
app = FastAPI(lifespan=lifespan)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 设置模板
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

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

# RTSP断开连接路由
@app.post("/api/rtsp/disconnect")
@api_error_handler(status_code=500)
async def disconnect_rtsp_endpoint():
    """
    断开RTSP视频流连接 - API端点
    """
    await disconnect_rtsp()
    return create_api_response(
        status="success",
        message="成功断开RTSP连接",
        connected=False
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

# 主页路由
@app.get("/")
async def home(request: Request):
    """
    主页路由
    
    渲染主页模板
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
@api_error_handler(status_code=503)
async def health_check():
    """
    健康检查端点
    
    检查服务健康状态
    """
    # 检查Kimi API密钥是否配置
    kimi_api_key = os.environ.get("MOONSHOT_API_KEY", "")
    if not kimi_api_key and "kimi" in MODELS:
        kimi_api_key = MODELS.get("kimi", {}).get("api_key", "")
        
    kimi_status = "已配置" if kimi_api_key else "未配置"
    
    return create_api_response(
        status="success",
        message="服务运行正常",
        models_status={
            "minimax": "可用",
            "qwen": "可用",
            "gemini": "可用",
            "kimi": kimi_status
        }
    )

@app.get("/test-kimi")
async def test_kimi():
    """
    测试Kimi模型的基本功能
    """
    # 检查API密钥是否配置
    api_key = os.environ.get("MOONSHOT_API_KEY", "")
    if not api_key and "kimi" in MODELS:
        api_key = MODELS.get("kimi", {}).get("api_key", "")
    
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "未设置Moonshot API密钥，请在环境变量或配置文件中设置MOONSHOT_API_KEY",
                "timestamp": time.time()
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "Kimi配置有效，API密钥已设置",
            "model": "moonshot-v1-8k-vision-preview",
            "timestamp": time.time()
        }
    )

# 图像分析API端点
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
    "kimi": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。",
    "douban": "你是一个专业的视觉分析助手，请帮助用户分析图像中的内容。请始终使用有效的JSON格式返回分析结果，确保每个目标都有明确的found属性，值为true或false。"
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
        从文本中提取和解析JSON内容，增强版本
        
        Args:
            content: 包含可能JSON内容的文本
            
        Returns:
            解析后的JSON对象，或None
        """
        if not content:
            return None
            
        # 多种解析策略
        strategies = [
            # 策略1: 提取```json```代码块中的内容 
            lambda text: re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL),
            
            # 策略2: 提取{}括起来的完整JSON
            lambda text: re.search(r'({.*})', text, re.DOTALL),
            
            # 策略3: 查找最长的{...}块
            lambda text: max(re.finditer(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', text), 
                            key=lambda m: len(m.group(0)), default=None)
        ]
        
        for strategy in strategies:
            try:
                match = strategy(content)
                if match:
                    json_text = match.group(1).strip()
                    
                    # 修复常见的格式问题
                    json_text = json_text.replace("'", '"')  # 单引号替换为双引号
                    json_text = re.sub(r',\s*}', '}', json_text)  # 移除末尾逗号
                    
                    # 尝试解析
                    parsed = json.loads(json_text)
                    logger.info(f"成功解析JSON: {json_text[:100]}...")
                    return parsed
            except (json.JSONDecodeError, AttributeError, KeyError, IndexError) as e:
                logger.debug(f"JSON解析策略失败: {str(e)}")
                continue
        
        # 最后的尝试：直接解析整个内容
        try:
            # 清理可能的非JSON文本，尝试获得一个有效的JSON字符串
            json_text = content
            json_text = re.sub(r'^[^{]*', '', json_text)  # 移除开头直到第一个{
            json_text = re.sub(r'[^}]*$', '', json_text)  # 移除最后一个}之后的所有内容
            
            # 尝试解析
            parsed = json.loads(json_text)
            logger.info(f"成功解析清理后的JSON: {json_text[:100]}...")
            return parsed
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.warning(f"所有JSON解析尝试均失败: {str(e)}")
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
    async def process_kimi(image_data, targets, system_message, user_message, start_time):
        """处理Moonshot Kimi模型的API调用"""
        try:
            # 获取Moonshot Kimi配置 - 从环境变量或配置中获取
            api_key = os.environ.get("MOONSHOT_API_KEY", "")
            if not api_key and "kimi" in MODELS:
                api_key = MODELS.get("kimi", {}).get("api_key", "")
            
            if not api_key:
                return format_api_response(
                    None, 
                    targets, 
                    error="未设置Moonshot API密钥，请在环境变量或配置文件中设置MOONSHOT_API_KEY", 
                    model_name="kimi"
                )
            
            # 提取消息内容
            text_content = await ModelProcessor._extract_user_message_text(user_message)
            if not text_content:
                text_content = await ModelProcessor._generate_default_query(targets)
                logger.warning(f"无法从user_message提取文本内容，使用默认查询: {text_content}")
            
            system_content = await ModelProcessor._extract_system_message_text(system_message)
            
            # 记录API调用开始
            api_start_time = time.time()
            logger.info(f"开始调用Kimi模型API，时间: {api_start_time}")
            
            # 初始化OpenAI客户端，使用Moonshot的API
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1"
            )
            
            # 发送请求
            response = client.chat.completions.create(
                model="moonshot-v1-8k-vision-preview",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        {"type": "text", "text": text_content}
                    ]}
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            # 计算API调用时间
            api_call_time = time.time() - api_start_time
            logger.info(f"模型Kimi API调用完成，耗时: {api_call_time:.2f}秒")
            
            # 提取响应内容
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                logger.debug(f"Kimi API 原始返回: {content}")
                
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
                            "model": "kimi",
                            "status": "success",
                            "timestamp": time.time()  # 添加时间戳确保排序
                        }
                
                # 使用通用响应解析
                return format_api_response(content, targets, model_name="kimi")
            else:
                # 响应没有有效的choices
                error_msg = "API响应缺少有效的choices"
                logger.warning(f"Kimi API 响应无效: {error_msg}")
                content = f"API响应格式无效: {str(response)}"
                
            # 使用标准格式化返回结果
            return format_api_response(content, targets, model_name="kimi")
                
        except Exception as api_error:
            logger.error(f"调用模型Kimi API时出错: {str(api_error)}")
            traceback_info = traceback.format_exc()
            logger.debug(f"调用Kimi出错详细信息: {traceback_info}")
            return format_api_response(None, targets, error=f"{str(api_error)}", model_name="kimi")
    
    @staticmethod
    async def process_douban(image_data, targets, system_message, user_message, start_time):
        """处理火山引擎豆包大模型的API调用"""
        try:
            # 获取豆包配置 - 从环境变量或配置中获取
            api_key = os.environ.get("ARK_API_KEY", "")
            if not api_key and "douban" in MODELS:
                api_key = MODELS.get("douban", {}).get("api_key", "")
            
            if not api_key:
                return format_api_response(
                    None, 
                    targets, 
                    error="未设置豆包 API密钥，请在环境变量或配置文件中设置ARK_API_KEY", 
                    model_name="douban"
                )
            
            # 提取消息内容
            text_content = await ModelProcessor._extract_user_message_text(user_message)
            if not text_content:
                text_content = await ModelProcessor._generate_default_query(targets)
                logger.warning(f"无法从user_message中提取文本内容，将使用生成的默认查询：{text_content}")
            
            # 增强提示词，要求返回标准化的JSON格式
            text_content += "\n\n请使用标准JSON格式返回分析结果，格式如下：\n```json\n{\n  \"description\": \"图像的整体描述\",\n  \"targets\": [\n    {\"name\": \"目标1\", \"found\": true},\n    {\"name\": \"目标2\", \"found\": false}\n  ]\n}\n```"
            
            # 获取系统提示词
            system_content = await ModelProcessor._extract_system_message_text(system_message)
            
            # 使用volcengine豆包SDK调用API
            from volcenginesdkarkruntime import Ark
            
            # 初始化客户端 - 按官方示例实现
            base_url = MODELS.get("douban", {}).get("base_url", "https://ark.cn-beijing.volces.com/api/v3")
            client = Ark(
                base_url=base_url,
                api_key=api_key
            )
            
            # 构建完整的消息列表
            messages = []
            # 添加系统消息（如果有）
            if system_content:
                messages.append({"role": "system", "content": system_content})
            
            # 添加用户消息
            messages.append({
                "role": "user", 
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": text_content}
                ]
            })
            
            # 使用模型配置中的模型名称，或使用默认模型
            model_name = MODELS.get("douban", {}).get("name", "doubao-1-5-vision-pro-32k-250115")
            logger.info(f"使用豆包模型: {model_name}")
            
            # 发送请求
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            
            # 解析响应
            result = {}
            if hasattr(response, 'choices') and len(response.choices) > 0 and hasattr(response.choices[0], 'message'):
                content = response.choices[0].message.content
                logger.info(f"豆包API原始响应: {content}")
                
                # 尝试解析JSON返回 - 与Qwen类似
                parsed_json = await ModelProcessor._parse_json_from_text(content)
                
                if parsed_json is not None:
                    # 提取关键信息
                    description = parsed_json.get("description", content)
                    parsed_targets = parsed_json.get("targets", [])
                    
                    # 如果成功解析了targets
                    if parsed_targets:
                        response_dict = {
                            "description": description,
                            "targets": parsed_targets,
                            "model": "douban",
                            "status": "success",
                            "timestamp": time.time()  # 添加时间戳确保排序
                        }
                        # 添加处理时间
                        processing_time = time.time() - start_time
                        response_dict["processing_time"] = processing_time
                        return response_dict
                
                # 如果无法解析为JSON，使用通用响应解析
                response_dict = format_api_response(content, targets, model_name="douban")
                # 添加处理时间
                processing_time = time.time() - start_time
                response_dict["processing_time"] = processing_time
                return response_dict
            else:
                logger.error(f"豆包API响应格式异常: {response}")
                return format_api_response(
                    None, 
                    targets, 
                    error=f"API响应格式异常: {response}", 
                    model_name="douban"
                )
            
        except Exception as e:
            logger.error(f"豆包API调用出错: {str(e)}")
            logger.error(traceback.format_exc())
            api_error = str(e)
            return format_api_response(None, targets, error=f"{str(api_error)}", model_name="douban")
    
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
        elif model_name == "kimi":
            return await cls.process_kimi(image_data, targets, system_message, user_message, start_time)
        elif model_name == "douban":
            # 检查volcengine豆包SDK是否可用
            if not ARK_IMPORTED:
                return format_api_response(
                    None, 
                    targets, 
                    error="火山引擎豆包SDK未安装，请执行 pip install volcengine-python-sdk", 
                    model_name="douban"
                )
            return await cls.process_douban(image_data, targets, system_message, user_message, start_time)
        else:
            return await cls.process_unsupported(model_name, targets, start_time)

if __name__ == "__main__":
    import uvicorn
    import sys
    import platform
    
    # 打印启动信息
    print("=" * 50)
    print(f"启动 Video Analysis 应用")
    print(f"Python 版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"工作目录: {os.getcwd()}")
    print(f"模板目录: {TEMPLATES_DIR}")
    print(f"静态文件目录: {STATIC_DIR}")
    print("=" * 50)
    
    # 检查必要的目录
    for directory in [TEMPLATES_DIR, STATIC_DIR]:
        if not directory.exists():
            print(f"警告: 目录不存在，将创建: {directory}")
            directory.mkdir(exist_ok=True)
    
    # 启动应用
    print("正在启动应用...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info" if not APP_CONFIG.get("debug", False) else "debug"
    )
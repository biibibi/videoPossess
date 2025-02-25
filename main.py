# main.py
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
import os
import cv2
import ollama
import time
from pathlib import Path
import asyncio
import json
import httpx
from ollama import Client
from contextlib import asynccontextmanager
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from models.model_manager import ModelManager

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ollama 客户端创建函数
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def create_ollama_client():
    """
    创建并初始化 Ollama 客户端
    包含连接测试、模型验证和基础功能测试
    使用重试机制处理临时性故障
    """
    try:
        # 测试服务器连接
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                raise Exception(f"服务器检查失败，状态码: {response.status_code}")
            
            # 验证模型可用性
            try:
                data = response.json()
                print("收到的模型数据:", data)
                
                # 检查返回的模型列表格式并验证所需模型是否存在
                if 'models' in data and isinstance(data['models'], list):
                    model_names = [m.get('name', '') for m in data['models']]
                    if 'llama3.2-vision:latest' not in model_names:
                        raise Exception("未找到所需模型 'llama3.2-vision:latest'")
                    print(f"在可用模型中找到目标模型: {model_names}")
                else:
                    raise Exception("API 响应格式异常")
                
            except json.JSONDecodeError as e:
                raise Exception(f"API 响应解析失败: {str(e)}")
            
            print(f"成功验证 Ollama 服务器和模型可用性")
            
            # 创建 Ollama 客户端实例
            client = Client(host=OLLAMA_BASE_URL)
            
            # 测试模型的基本对话功能
            test_response = await asyncio.to_thread(
                client.chat,
                model='llama3.2-vision:latest',
                messages=[{
                    'role': 'user',
                    'content': '你好'
                }]
            )
            
            # 验证响应格式
            if test_response and 'message' in test_response:
                print("成功测试模型对话功能")
                print(f"测试响应: {test_response['message'].get('content', '')[:100]}...")
                return client
            else:
                raise Exception("模型测试失败: 响应格式无效")
                
    except httpx.RequestError as e:
        print(f"网络错误: {str(e)}")
        raise
    except Exception as e:
        print(f"客户端创建错误: {str(e)}")
        raise

# 全局 Ollama 客户端
ollama_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用程序生命周期管理
    处理启动时的客户端初始化和关闭时的资源清理
    """
    global ollama_client
    retry_count = 0
    max_retries = 3
    
    # 尝试初始化客户端，最多重试3次
    while retry_count < max_retries:
        try:
            ollama_client = await create_ollama_client()
            if ollama_client:
                print("成功初始化 Ollama 客户端")
                break
        except Exception as e:
            retry_count += 1
            print(f"第 {retry_count}/{max_retries} 次尝试失败: {str(e)}")
            if retry_count < max_retries:
                await asyncio.sleep(5)  # 失败后等待5秒再重试
    
    try:
        yield
    finally:
        print("清理资源...")

# 创建 FastAPI 应用实例
app = FastAPI(lifespan=lifespan)

# 创建必要的目录
UPLOAD_DIR = Path("uploads")
FRAMES_DIR = Path("frames")
UPLOAD_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

# 配置模板和静态文件服务
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

# Ollama 服务器地址
OLLAMA_BASE_URL = "http://123.157.129.172:3336"

def ensure_directories():
    """确保必要的目录存在且具有正确的权限"""
    directories = [
        Path("uploads"),
        Path("frames"),
        Path("static"),
        Path("templates")
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        # 确保目录具有写入权限
        os.chmod(directory, 0o755)
        print(f"目录已创建/确认: {directory}")

# 在应用启动时调用
@app.on_event("startup")
async def startup_event():
    ensure_directories()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def preprocess_image(image_path):
    """图像预处理函数"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False

        # 对比度增强
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 保存处理后的图像
        cv2.imwrite(image_path, final, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return True
    except Exception as e:
        logger.error(f"图像预处理失败: {str(e)}")
        return False

@app.post("/analyze")
async def analyze_video(
    request: Request,
    video: UploadFile = File(...),
    object_str: str = Form(...),
    model: str = Form(default="llama")
):
    try:
        logger.info("=== 开始处理视频分析请求 ===")
        logger.info(f"请求头信息: {dict(video.headers) if video and video.headers else 'None'}")
        logger.info(f"文件信息: 名称={video.filename if video else 'None'}, 类型={video.content_type if video else 'None'}")
        logger.info(f"搜索对象: {object_str}")
        logger.info(f"选择的模型: {model}")

        # 文件验证
        if not video or not video.file:
            raise HTTPException(status_code=400, detail="未接收到视频文件")

        # 检查文件类型
        content_type = video.content_type or ''
        if not any(media_type in content_type.lower() for media_type in ['video/', 'application/octet-stream']):
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {content_type}")

        # 保存上传的视频
        video_path = UPLOAD_DIR / video.filename
        try:
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            logger.info(f"视频保存成功: {video_path}")
        except Exception as e:
            logger.error(f"保存视频文件失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"保存视频文件失败: {str(e)}")

        # 为当前任务创建专门的帧目录
        task_frames_dir = FRAMES_DIR / video.filename.split('.')[0]
        task_frames_dir.mkdir(exist_ok=True)
        logger.info(f"创建帧目录: {task_frames_dir}")

        # 获取模型处理器
        model_handler = ModelManager.get_model_handler(model)
        if not model_handler:
            raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model}")

        async def generate_results():
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise Exception("无法打开视频文件")

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_second = int(frame_count / fps)
                    frame_path = task_frames_dir / f"frame_{current_second}.jpg"
                    
                    # 每秒保存一帧
                    if frame_count % int(fps) == 0:
                        cv2.imwrite(str(frame_path), frame)
                        logger.debug(f"保存帧: {frame_path}")

                        if preprocess_image(str(frame_path)):
                            try:
                                is_match, description, confidence = await model_handler.analyze_image(str(frame_path), object_str)
                                
                                result = {
                                    "status": "success",
                                    "frame": {
                                        "second": current_second,
                                        "is_match": is_match,
                                        "description": description,
                                        "confidence": confidence,
                                        "frame_path": f"/frames/{video.filename.split('.')[0]}/frame_{current_second}.jpg"
                                    }
                                }
                                yield json.dumps(result) + "\n"
                            except Exception as e:
                                logger.error(f"分析帧时出错: {str(e)}")
                                yield json.dumps({
                                    "status": "error",
                                    "message": f"分析帧时出错: {str(e)}"
                                }) + "\n"

                    frame_count += 1

                cap.release()
                logger.info("视频处理完成")

            except Exception as e:
                logger.error(f"生成结果时出错: {str(e)}")
                yield json.dumps({
                    "status": "error",
                    "message": f"生成结果时出错: {str(e)}"
                }) + "\n"

        return StreamingResponse(generate_results(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 修改测试连接函数
async def test_ollama_connection():
    """
    测试与 Ollama 服务器的连接
    返回布尔值表示连接是否成功
    """
    try:
        timeout = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                print(f"成功连接到 Ollama 服务器: {OLLAMA_BASE_URL}")
                return True
            print(f"连接到 Ollama 服务器失败。状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"连接 Ollama 服务器时发生错误: {str(e)}")
        return False

async def check_client_status():
    """
    检查客户端状态并在需要时尝试重新初始化
    返回布尔值表示客户端是否可用
    """
    global ollama_client
    
    if ollama_client is None:
        try:
            ollama_client = await create_ollama_client()
            return True
        except Exception as e:
            print(f"重新初始化客户端失败: {str(e)}")
            return False
    return True

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        if not await check_client_status():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "错误",
                    "message": "Ollama 客户端初始化失败"
                }
            )
        
        try:
            response = await asyncio.to_thread(
                ollama_client.chat,
                model='llama3.2-vision:latest',
                messages=[{
                    'role': 'user',
                    'content': '测试消息'
                }]
            )
            return {
                "status": "正常",
                "ollama_server": "已连接",
                "model_status": "可用"
            }
        except Exception as model_error:
            print(f"模型测试错误: {str(model_error)}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "错误",
                    "message": f"模型测试失败: {str(model_error)}"
                }
            )
    except Exception as e:
        print(f"健康检查错误: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "错误",
                "message": str(e)
            }
        )

@app.get("/api/test")
async def test_api():
    """API 测试端点"""
    results = {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            tags = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            results["available_models"] = tags.json() if tags.status_code == 200 else None
            
            results["server_info"] = {
                "url": OLLAMA_BASE_URL,
                "connection_status": "connected" if tags.status_code == 200 else "failed",
                "status_code": tags.status_code
            }

            return {
                "status": "success",
                "results": results
            }
        except Exception as e:
            print(f"API 测试错误: {str(e)}")
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
    try:
        response = await asyncio.to_thread(
            ollama_client.chat,
            model='llama3.2-vision:latest',
            messages=[{
                'role': 'user',
                'content': '请简单介绍一下你自己'
            }]
        )
        return {"status": "成功", "response": response}
    except Exception as e:
        print(f"测试错误: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "错误",
                "message": str(e)
            }
        )

@app.post("/test-vision")
async def test_vision(request: Request):
    """测试视觉模型的简单端点"""
    try:
        response = await asyncio.to_thread(
            ollama_client.chat,
            model='llama3.2-vision:latest',
            messages=[{
                'role': 'user',
                'content': '这是一个测试消息，请确认你可以正常工作。'
            }]
        )
        return {
            "status": "success",
            "model": "llama3.2-vision:latest",
            "response": response
        }
    except Exception as e:
        print(f"Vision test error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
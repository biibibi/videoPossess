# Gemini API 使用示例

本文档提供了在视频分析工具中使用 Google Gemini API 的具体示例和最佳实践。

## 1. 简单的图像分析调用

以下是通过 HTTP API 调用 Gemini 分析图像的基本方式:

```javascript
// 前端代码示例
async function analyzeImage() {
  // 获取base64编码的图像数据
  const imageBase64 = getImageBase64(); // 根据实际情况获取
  
  // 要搜索的目标
  const targets = ["人物", "电脑", "书籍"];
  
  // 调用分析API
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      image: imageBase64,
      targets: targets,
      model: "gemini"  // 指定使用Gemini模型
    })
  });
  
  const result = await response.json();
  console.log('分析结果：', result);
  
  // 处理结果
  if (result.status === "success") {
    // 显示描述
    document.getElementById('description').textContent = result.description;
    
    // 显示目标检测结果
    const targetsList = document.getElementById('targets-list');
    targetsList.innerHTML = '';
    result.targets.forEach(target => {
      const li = document.createElement('li');
      li.textContent = `${target.name}: ${target.found ? '找到' : '未找到'}`;
      li.className = target.found ? 'found' : 'not-found';
      targetsList.appendChild(li);
    });
  }
}
```

## 2. 直接在Python代码中使用

在Python后端代码中直接使用Gemini模型进行分析:

```python
import asyncio
from main import ModelProcessor
import base64
import time
import cv2

async def analyze_frame_with_gemini(frame, targets=None):
    """使用Gemini模型分析视频帧"""
    if targets is None:
        targets = ["人物", "电脑", "文件", "手机"]
    
    # 将帧转换为base64图像
    success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        return {"error": "无法编码图像"}
        
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # 获取当前时间作为开始时间
    start_time = time.time()
    
    # 调用模型处理器
    result = await ModelProcessor.process(
        model_name="gemini", 
        image=image_base64, 
        targets=targets, 
        start_time=start_time
    )
    
    return result

# 示例使用：分析单个图像
async def test_image_analysis():
    # 读取图像
    image = cv2.imread('path/to/your/image.jpg')
    if image is None:
        print("无法读取图像")
        return
    
    # 分析图像
    result = await analyze_frame_with_gemini(image)
    
    # 打印结果
    print(f"描述: {result.get('description', '无描述')}")
    print("目标检测结果:")
    for target in result.get('targets', []):
        print(f"  {target['name']}: {'找到' if target['found'] else '未找到'}")
    
    return result

# 示例使用：分析视频帧
async def analyze_video_frames(video_path, interval=30):
    """
    分析视频中的帧，每隔interval帧分析一次
    
    Args:
        video_path: 视频文件路径
        interval: 帧间隔，每隔多少帧分析一次
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []
    
    results = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔interval帧分析一次
        if frame_count % interval == 0:
            print(f"分析第 {frame_count} 帧...")
            result = await analyze_frame_with_gemini(frame)
            
            # 添加帧索引到结果
            result['frame_index'] = frame_count
            results.append(result)
        
        frame_count += 1
    
    cap.release()
    print(f"视频分析完成，共分析 {len(results)} 个帧")
    return results

# 运行示例
if __name__ == "__main__":
    asyncio.run(test_image_analysis())
    # 或者分析视频
    # results = asyncio.run(analyze_video_frames("path/to/your/video.mp4", interval=30))
```

## 3. 智能分析模式用法

除了常规的目标检测外，Gemini 模型还支持智能分析模式，可以回答关于图像的具体问题:

```python
async def ask_about_image(image_path, question):
    """使用Gemini模型回答关于图像的问题"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "无法读取图像"}
    
    # 将帧转换为base64图像
    success, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        return {"error": "无法编码图像"}
        
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # 设置智能分析模式
    # 第一个元素是问题，第二个元素是模式标记
    targets = [question, "mode=intelligent_analysis"]
    
    # 调用模型处理器
    result = await ModelProcessor.process(
        model_name="gemini", 
        image=image_base64, 
        targets=targets, 
        start_time=time.time()
    )
    
    return result

# 示例使用
async def test_intelligent_analysis():
    # 调用智能分析
    result = await ask_about_image(
        "path/to/meeting_room.jpg",
        "这个会议室里有多少人？他们在做什么？"
    )
    
    # 打印结果
    print(f"回答: {result.get('description', '无回答')}")
    
    return result

# 运行示例
if __name__ == "__main__":
    asyncio.run(test_intelligent_analysis())
```

## 4. 最佳实践

1. **代理设置**: 在中国大陆使用时，确保正确配置代理设置:
   
   ```python
   # 在config.py中
   GEMINI_CONFIG = {
       # 其他配置...
       "proxy": {
           "http": "http://你的代理服务器:端口",
           "https": "http://你的代理服务器:端口",
       }
   }
   ```

2. **错误处理**: 确保处理API调用可能出现的错误:
   
   ```python
   try:
       result = await ModelProcessor.process("gemini", image_base64, targets, time.time())
       # 处理成功结果
   except Exception as e:
       # 处理错误
       print(f"API调用失败: {e}")
       # 可以尝试降级到另一个模型
       result = await ModelProcessor.process("minimax", image_base64, targets, time.time())
   ```

3. **降级策略**: 如果Gemini API不可用，可以设置自动降级到其他模型:
   
   ```python
   async def analyze_with_fallback(image, targets):
       """带有降级策略的分析函数"""
       models = ["gemini", "minimax", "qwen"]  # 按优先级排序的模型
       
       for model in models:
           try:
               result = await ModelProcessor.process(model, image, targets, time.time())
               if result.get("status") == "success":
                   return result
           except Exception as e:
               print(f"模型 {model} 调用失败: {e}")
               continue
       
       # 所有模型都失败时返回的结果
       return {
           "status": "error", 
           "message": "所有模型调用都失败",
           "targets": [{"name": t, "found": False} for t in targets]
       }
   ```

## 5. 性能优化建议

1. **图像压缩**: 在上传图像前进行适当压缩，减少传输时间和API调用成本:
   
   ```python
   def optimize_image(image, max_size=(800, 800), quality=85):
       """优化图像尺寸和质量"""
       h, w = image.shape[:2]
       
       # 调整尺寸
       if h > max_size[0] or w > max_size[1]:
           # 保持宽高比
           if h > w:
               new_h = max_size[0]
               new_w = int(w * (new_h / h))
           else:
               new_w = max_size[1]
               new_h = int(h * (new_w / w))
           
           image = cv2.resize(image, (new_w, new_h))
       
       # 编码为JPEG，控制质量
       params = [cv2.IMWRITE_JPEG_QUALITY, quality]
       _, buffer = cv2.imencode('.jpg', image, params)
       
       return buffer
   ```

2. **并行处理**: 处理多个图像时使用并行调用:
   
   ```python
   async def analyze_multiple_images(image_paths, targets):
       """并行分析多张图像"""
       tasks = []
       
       for path in image_paths:
           image = cv2.imread(path)
           if image is None:
               continue
               
           # 编码图像
           success, buffer = cv2.imencode('.jpg', image)
           if not success:
               continue
               
           image_base64 = base64.b64encode(buffer).decode('utf-8')
           
           # 创建任务
           task = ModelProcessor.process("gemini", image_base64, targets, time.time())
           tasks.append(task)
       
       # 并行执行所有任务
       results = await asyncio.gather(*tasks, return_exceptions=True)
       
       # 处理结果
       processed_results = []
       for i, result in enumerate(results):
           if isinstance(result, Exception):
               print(f"图像 {image_paths[i]} 处理失败: {result}")
               processed_results.append({
                   "status": "error",
                   "message": str(result),
                   "image_path": image_paths[i]
               })
           else:
               result["image_path"] = image_paths[i]
               processed_results.append(result)
       
       return processed_results
   ```
``` 
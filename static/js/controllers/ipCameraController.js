// IP摄像头控制器
window.ipCameraController = {
    // 帧获取计时器
    frameTimer: null,
    // 帧更新间隔（毫秒）
    frameInterval: 100,
    // 分析相关变量
    analysisActive: false,
    frameBuffer: [],
    lastAnalysisTime: 0,
    currentFrameIndex: 0,
    maxBufferSize: 3,
    
    /**
     * 连接到RTSP摄像头
     * @param {string} rtspUrl - RTSP地址
     */
    async connectToCamera(rtspUrl) {
        try {
            const streamContainer = document.querySelector('#ipCameraSection .stream-container');
            
            if (!rtspUrl) {
                throw new Error('请输入有效的摄像头地址');
            }
            
            // 清空容器
            streamContainer.innerHTML = '';
            
            // 创建占位符和视频元素
            const placeholder = document.createElement('div');
            placeholder.className = 'ipCamera-placeholder';
            placeholder.innerHTML = '<i class="bi bi-arrow-repeat spin"></i><p>正在连接RTSP摄像头...</p>';
            streamContainer.appendChild(placeholder);
            
            // 更新按钮状态
            if (elements.connectCameraBtn) {
                elements.connectCameraBtn.disabled = true;
                elements.connectCameraBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> 连接中...';
            }
            
            // 发送连接请求到后端
            const response = await fetch('/api/rtsp/connect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: rtspUrl })
            });
            
            const data = await response.json();
            
            if (!response.ok || data.status === 'error') {
                throw new Error(data.message || '连接失败');
            }
            
            // 连接成功，使用MJPEG流
            placeholder.style.display = 'none';
            
            // 创建img元素用于显示MJPEG流
            const img = document.createElement('img');
            img.id = 'ipCameraStream';
            img.style.width = '100%';
            img.style.height = '100%';
            img.style.objectFit = 'contain';
            // 添加时间戳参数以防止缓存
            img.src = `/api/rtsp/stream?t=${new Date().getTime()}`;
            streamContainer.appendChild(img);
            
            // 处理图像加载错误
            img.onerror = () => {
                console.error('RTSP流加载失败');
                this.handleStreamError();
            };
            
            // 更新状态和UI
            state.ipCameraActive = true;
            state.ipCameraUrl = rtspUrl;
            
            // 更新按钮状态
            if (elements.connectCameraBtn) {
                elements.connectCameraBtn.disabled = false;
                elements.connectCameraBtn.innerHTML = '<i class="bi bi-stop-fill"></i> 断开';
                elements.connectCameraBtn.classList.remove('btn-primary');
                elements.connectCameraBtn.classList.add('btn-danger');
            }
            
            // 显示成功消息
            uiController.showAlert('info', '成功连接到RTSP摄像头');
            
            // 启动心跳检查，确保连接仍然有效
            this.startHeartbeat();
            
        } catch (error) {
            console.error('连接RTSP摄像头时出错:', error);
            
            // 更新按钮状态
            if (elements.connectCameraBtn) {
                elements.connectCameraBtn.disabled = false;
                elements.connectCameraBtn.innerHTML = '<i class="bi bi-play-fill"></i> 连接';
            }
            
            // 显示失败占位符
            const streamContainer = document.querySelector('#ipCameraSection .stream-container');
            streamContainer.innerHTML = '';
            
            const placeholder = document.createElement('div');
            placeholder.className = 'ipCamera-placeholder';
            placeholder.innerHTML = `
                <i class="bi bi-exclamation-triangle"></i>
                <p>连接失败: ${error.message}</p>
                <p class="mt-2 text-muted">请检查RTSP地址并重试</p>
            `;
            
            streamContainer.appendChild(placeholder);
            
            // 显示错误消息
            uiController.showAlert('error', '连接失败: ' + error.message);
        }
    },
    
    /**
     * 处理流错误
     */
    handleStreamError() {
        // 如果已经断开连接，不做处理
        if (!state.ipCameraActive) return;
        
        const img = document.getElementById('ipCameraStream');
        if (img) {
            // 尝试重新加载流
            setTimeout(() => {
                // 添加时间戳参数以防止缓存
                img.src = `/api/rtsp/stream?t=${new Date().getTime()}`;
            }, 2000);
            
            // 计数错误次数
            this.errorCount = (this.errorCount || 0) + 1;
            
            // 如果连续错误超过阈值，断开连接
            if (this.errorCount > 5) {
                console.error('RTSP流连续加载失败，断开连接');
                this.disconnectCamera();
                uiController.showAlert('error', 'RTSP流连接失败，已断开');
            }
        }
    },
    
    /**
     * 启动心跳检查
     */
    startHeartbeat() {
        // 停止之前可能存在的心跳
        this.stopHeartbeat();
        
        // 每10秒检查一次连接状态
        this.heartbeatTimer = setInterval(async () => {
            try {
                const isConnected = await this.checkConnectionStatus();
                if (!isConnected && state.ipCameraActive) {
                    console.warn('RTSP连接已断开');
                    
                    // 重新加载流试图恢复连接
                    const img = document.getElementById('ipCameraStream');
                    if (img) {
                        img.src = `/api/rtsp/stream?t=${new Date().getTime()}`;
                    }
                    
                    // 计数错误次数
                    this.heartbeatErrorCount = (this.heartbeatErrorCount || 0) + 1;
                    
                    // 如果连续错误超过阈值，断开连接
                    if (this.heartbeatErrorCount > 3) {
                        console.error('RTSP连接心跳检查失败，断开连接');
                        this.disconnectCamera();
                        uiController.showAlert('error', 'RTSP流连接已丢失，已断开');
                    }
                } else {
                    // 重置错误计数
                    this.heartbeatErrorCount = 0;
                }
            } catch (error) {
                console.error('心跳检查出错:', error);
            }
        }, 10000);
    },
    
    /**
     * 停止心跳检查
     */
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
        this.heartbeatErrorCount = 0;
    },
    
    /**
     * 断开摄像头连接
     */
    async disconnectCamera() {
        try {
            // 停止帧获取和心跳检查
            this.stopFrameCapture();
            this.stopHeartbeat();
            
            // 发送断开请求到后端
            const response = await fetch('/api/rtsp/disconnect', {
                method: 'POST'
            });
            
            // 更新状态
            state.ipCameraActive = false;
            state.ipCameraUrl = null;
            
            // 更新按钮状态
            if (elements.connectCameraBtn) {
                elements.connectCameraBtn.disabled = false;
                elements.connectCameraBtn.innerHTML = '<i class="bi bi-play-fill"></i> 连接';
                elements.connectCameraBtn.classList.remove('btn-danger');
                elements.connectCameraBtn.classList.add('btn-primary');
            }
            
            // 显示断开占位符
            const streamContainer = document.querySelector('#ipCameraSection .stream-container');
            streamContainer.innerHTML = '';
            
            const placeholder = document.createElement('div');
            placeholder.className = 'ipCamera-placeholder';
            placeholder.innerHTML = '<i class="bi bi-webcam"></i><p>已断开连接，请输入RTSP地址并点击连接</p>';
            
            streamContainer.appendChild(placeholder);
            
            // 显示消息
            uiController.showAlert('info', '已断开RTSP连接');
            
        } catch (error) {
            console.error('断开RTSP连接时出错:', error);
            uiController.showAlert('error', '断开连接时出错: ' + error.message);
        }
    },
    
    /**
     * 停止帧捕获
     */
    stopFrameCapture() {
        if (this.frameTimer) {
            clearInterval(this.frameTimer);
            this.frameTimer = null;
        }
        
        // 重置错误计数
        this.errorCount = 0;
    },
    
    /**
     * 获取当前视频帧（用于分析）
     * @returns {Promise<string|null>} base64编码的图像数据或null
     */
    async getCurrentFrame() {
        try {
            if (!state.ipCameraActive) {
                return null;
            }
            
            // 直接从后端获取当前帧
            const response = await fetch('/api/rtsp/frame');
            
            if (!response.ok) {
                return null;
            }
            
            const data = await response.json();
            
            if (data.status === 'error' || !data.frame) {
                return null;
            }
            
            return data.frame;
            
        } catch (error) {
            console.error('获取当前帧时出错:', error);
            return null;
        }
    },
    
    /**
     * 检查RTSP连接状态
     * @returns {Promise<boolean>} 连接状态
     */
    async checkConnectionStatus() {
        try {
            const response = await fetch('/api/rtsp/status');
            
            if (!response.ok) {
                return false;
            }
            
            const data = await response.json();
            return data.connected === true;
            
        } catch (error) {
            console.error('检查RTSP状态时出错:', error);
            return false;
        }
    },
    
    /**
     * 捕获当前帧 - 供分析模块使用
     * 该方法与本地摄像头的captureCurrentFrame保持一致的接口
     * @returns {string|null} base64编码的图像数据或null
     */
    captureCurrentFrame() {
        // 使用缓存的截图，而不是实时获取
        // 这样避免了每次分析都要发起HTTP请求
        if (!state.ipCameraActive) {
            return null;
        }
        
        // 从页面上获取最新的RTSP图像
        const streamImg = document.getElementById('ipCameraStream');
        if (!streamImg || !streamImg.complete || streamImg.naturalWidth === 0) {
            console.error('网络摄像头图像不可用');
            return null;
        }
        
        try {
            // 使用canvas捕获当前图像
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = streamImg.naturalWidth;
            canvas.height = streamImg.naturalHeight;
            
            ctx.drawImage(streamImg, 0, 0, canvas.width, canvas.height);
            
            return canvas.toDataURL('image/jpeg', 0.85);
        } catch (error) {
            console.error('捕获网络摄像头帧时出错:', error);
            return null;
        }
    },
    
    /**
     * 开始网络摄像头分析
     * @param {Function} analysisCallback - 分析完成后的回调函数
     */
    startAnalysis(analysisCallback) {
        if (this.analysisActive) {
            return;
        }
        
        if (!state.ipCameraActive) {
            uiController.showAlert('error', '请先连接到网络摄像头');
            return;
        }
        
        // 初始化分析变量
        this.analysisActive = true;
        this.frameBuffer = [];
        this.lastAnalysisTime = 0;
        this.currentFrameIndex = 0;
        this.analysisCallback = analysisCallback;
        
        // 开始分析循环
        this.analyzeNextFrame();
        
        uiController.showAlert('info', '网络摄像头分析已开始');
    },
    
    /**
     * 停止网络摄像头分析
     */
    stopAnalysis() {
        if (!this.analysisActive) return;
        
        // 处理剩余帧
        if (this.frameBuffer.length > 0) {
            this.processBatchFrames();
        }
        
        this.analysisActive = false;
        uiController.showAlert('info', '网络摄像头分析已停止');
    },
    
    /**
     * 分析下一帧
     */
    async analyzeNextFrame() {
        try {
            if (!this.analysisActive || !state.ipCameraActive) {
                this.stopAnalysis();
                return;
            }
            
            // 获取当前帧
            const frameImage = this.captureCurrentFrame();
            if (!frameImage) {
                // 重试捕获帧
                setTimeout(() => this.analyzeNextFrame(), 500);
                return;
            }
            
            // 当前时间与上次分析时间的差值
            const timeSinceLastAnalysis = Date.now() - this.lastAnalysisTime;
            const frameInterval = 1000; // 固定为1秒间隔
            
            // 如果距离上次分析的时间不足帧间隔，则跳过当前帧
            if (this.lastAnalysisTime > 0 && timeSinceLastAnalysis < frameInterval) {
                // 计算下次分析前需要等待的时间
                const waitTime = Math.max(50, frameInterval - timeSinceLastAnalysis);
                setTimeout(() => this.analyzeNextFrame(), waitTime);
                return;
            }
            
            // 预处理图像 (如果需要)
            let processedImage = frameImage;
            if (window.imagePreprocessor && typeof imagePreprocessor.preprocessImage === 'function') {
                processedImage = await imagePreprocessor.preprocessImage(frameImage);
            }
            
            // 将当前帧添加到缓冲区
            this.frameBuffer.push({
                timestamp: Date.now(),
                frameNumber: this.currentFrameIndex + 1,
                image: processedImage
            });
            
            this.currentFrameIndex++;
            this.lastAnalysisTime = Date.now();
            
            // 当收集到足够帧或停止分析时处理缓冲区
            if (this.frameBuffer.length >= this.maxBufferSize || !this.analysisActive) {
                await this.processBatchFrames();
            }
            
            // 继续收集下一帧
            setTimeout(() => this.analyzeNextFrame(), 100);
            
        } catch (error) {
            console.error('分析网络摄像头帧时出错:', error);
            uiController.showAlert('error', '分析网络摄像头时出错: ' + error.message);
            this.stopAnalysis();
        }
    },
    
    /**
     * 批量处理帧
     */
    async processBatchFrames() {
        try {
            if (this.frameBuffer.length === 0) return;
            
            uiController.showAlert('info', `正在分析: ${this.frameBuffer.length}帧`);
            
            // 创建批处理任务
            const batchPromises = this.frameBuffer.map(async (frameData) => {
                // 调用分析回调函数
                const result = await this.analysisCallback(frameData.image, state.searchTargets);
                
                // 添加结果
                if (result && typeof searchResultController !== 'undefined' && 
                    result.targets && result.targets.length > 0) {
                    searchResultController.addResult({
                        timestamp: frameData.timestamp,
                        frameNumber: frameData.frameNumber,
                        frameImage: frameData.image,
                        description: result.description,
                        targets: result.targets,
                        videoTime: null // 网络摄像头没有时间戳
                    });
                }
                
                return result;
            });
            
            // 并行处理所有帧
            await Promise.all(batchPromises);
            
            // 更新UI
            if (typeof searchResultController !== 'undefined') {
                searchResultController.sortResults();
                searchResultController.renderResults();
            }
            
            // 清空缓冲区
            this.frameBuffer = [];
            
        } catch (error) {
            console.error('批量处理网络摄像头帧时出错:', error);
            this.frameBuffer = []; // 出错时也清空缓冲区
        }
    }
};

// 添加自动重连功能
setInterval(async () => {
    // 只有当标记为已连接但实际未连接时尝试重连
    if (state.ipCameraActive && !state.manualDisconnect) {
        const isConnected = await ipCameraController.checkConnectionStatus();
        
        if (!isConnected) {
            // 尝试重新连接
            if (state.ipCameraUrl) {
                ipCameraController.connectToCamera(state.ipCameraUrl);
            } else {
                // 没有URL，标记为断开
                state.ipCameraActive = false;
            }
        }
    }
}, 10000); // 每10秒检查一次

// 添加旋转动画样式
const style = document.createElement('style');
style.textContent = `
    .spin {
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);

// 确保ipCameraController在全局作用域中可用
window.ipCameraController = ipCameraController; 
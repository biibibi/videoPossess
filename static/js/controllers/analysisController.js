import CONFIG from '../config/config.js';
import elements from '../config/elements.js';
import state from '../config/state.js';
import utils from '../utils/utils.js';
import uiController from './uiController.js';
import searchResultController from './searchResultController.js';
import streamController from './streamController.js';

// 分析控制器
const analysisController = {
    maxRetries: 3, // 最大重试次数
    retryDelay: 1000, // 重试延迟(毫秒)
    frameCount: 0, // 已分析的帧数
    activeControllers: new Set(), // 跟踪活动的AbortController实例

    /**
     * 开始分析
     */
    async startAnalysis() {
        try {
            // 检查是否已在分析中
            if (state.isAnalyzing) {
                uiController.showAlert('分析正在进行中', 'info');
                return;
            }
            
            // 检查是否已添加搜索目标
            if (state.searchTargets.length === 0) {
                uiController.showAlert('请先添加搜索目标', 'error');
                return;
            }
            
            // 检查是否已上传视频、启动视频流或连接RTSP
            if (!state.hasUploadedVideo && !state.streamActive && !state.ipCameraActive) {
                uiController.showAlert('请先上传视频、启动本地摄像头或连接网络摄像头', 'error');
                return;
            }
            
            // 检查上传视频模式下是否已抽取帧
            if (state.hasUploadedVideo && (!state.extractedFrames || state.extractedFrames.length === 0)) {
                uiController.showAlert('视频帧抽取未完成', 'error');
                return;
            }
            
            // 初始化分析状态
            state.isAnalyzing = true;
            this.frameCount = 0;
            
            // 更新按钮状态
            this.updateAnalyzeButton('analyzing');
            
            // 清空之前的结果
            searchResultController.clearResults();
            
            // 根据模式选择不同的分析逻辑
            if (state.streamActive) {
                // 本地摄像头模式：定时分析当前帧
                await this.startStreamAnalysis();
            } else if (state.ipCameraActive) {
                // 网络摄像头模式：使用ipCameraController进行分析
                try {
                    // 使用ipCameraController中的方法直接开始分析
                    if (typeof ipCameraController.startAnalysis === 'function') {
                        ipCameraController.startAnalysis(this.analyzeFrame.bind(this));
                    } else {
                        // 如果没有startAnalysis方法，回退到自带的实现
                        console.warn('ipCameraController.startAnalysis方法不可用，使用内置分析逻辑');
                        await this.startIpCameraAnalysis();
                    }
                } catch (error) {
                    console.error('启动网络摄像头分析时出错:', error);
                    await this.startIpCameraAnalysis();
                }
            } else {
                // 上传视频模式：分析已抽取的帧
                await this.analyzeUploadedVideo();
            }
            
        } catch (error) {
            console.error('分析过程中出错:', error);
            uiController.showAlert(error.message || '分析过程中出错', 'error');
            this.stopAnalysis();
        }
    },
    
    /**
     * 更新分析按钮状态
     * @param {string} state - 按钮状态：'start'|'analyzing'
     * @param {number} [frameCount] - 已分析的帧数
     */
    updateAnalyzeButton(state, frameCount) {
        const btn = elements.analyzeBtn;
        if (!btn) return;
        
        // 移除所有可能的事件监听器
        const newBtn = btn.cloneNode(true);
        btn.parentNode.replaceChild(newBtn, btn);
        
        // 更新全局引用
        elements.analyzeBtn = newBtn;
        
        switch (state) {
            case 'start':
                newBtn.disabled = false;
                newBtn.innerHTML = '<i class="bi bi-play-fill"></i> 开始分析';
                newBtn.onclick = () => this.startAnalysis();
                break;
            case 'analyzing':
                newBtn.disabled = true;
                newBtn.innerHTML = frameCount 
                    ? `<span class="spinner-border spinner-border-sm"></span> 分析中 (${frameCount}帧)`
                    : '<span class="spinner-border spinner-border-sm"></span> 分析中...';
                break;
        }
    },
    
    /**
     * 开始视频流分析
     */
    async startStreamAnalysis() {
        try {
            uiController.showAlert('开始实时视频分析', 'info');
            
            // 更新按钮为分析中状态
            this.updateAnalyzeButton('analyzing');
            
            // 记录分析开始时间
            const startTime = Date.now();
            const maxAnalysisTime = 10 * 60 * 1000; // 最大分析时间10分钟
            
            // 立即分析第一帧
            const initialFrame = streamController.getCurrentFrame();
            if (initialFrame) {
                const initialTime = Date.now() / 1000;
                await this.analyzeFrame(initialFrame, initialTime);
                this.frameCount++;
                this.updateAnalyzeButton('analyzing', this.frameCount);
            } else {
                console.error('无法获取初始帧，请检查视频流是否正常');
                uiController.showAlert('无法获取初始帧，请检查视频流是否正常', 'error');
                return;
            }
            
            // 确保之前的定时器被清除
            if (state.streamFrameTimer) {
                cancelAnimationFrame(state.streamFrameTimer);
                state.streamFrameTimer = null;
            }
            
            // 使用更快的防抖时间
            const updateUI = utils.debounce(() => {
                this.updateAnalyzeButton('analyzing', this.frameCount);
            }, 150); // 降低防抖时间，使UI更新更及时
            
            // 创建分析队列，保证一次只处理一帧
            let isProcessing = false;
            let consecutiveErrors = 0;
            const maxConsecutiveErrors = 5; // 最大连续错误次数
            
            // 连续帧处理函数
            const processContinuously = async () => {
                // 检查是否超过最大分析时间
                const currentTime = Date.now();
                const elapsedTime = currentTime - startTime;
                
                if (elapsedTime >= maxAnalysisTime) {
                    uiController.showAlert(`已达到最大分析时间 ${maxAnalysisTime/1000} 秒，自动停止分析`, 'info');
                    this.stopAnalysis();
                    return;
                }
                
                // 检查视频流是否仍然活跃
                if (!state.streamActive) {
                    uiController.showAlert('视频流已停止，分析已结束', 'info');
                    this.stopAnalysis();
                    return;
                }
                
                if (!state.isAnalyzing) {
                    return;
                }
                
                // 如果当前没有正在处理的帧，获取并处理新帧
                if (!isProcessing) {
                    isProcessing = true;
                    
                    try {
                        // 获取当前帧
                        const frameData = streamController.getCurrentFrame();
                        if (!frameData) {
                            console.error('无法获取当前帧');
                            consecutiveErrors++;
                            
                            if (consecutiveErrors >= maxConsecutiveErrors) {
                                console.error(`连续 ${maxConsecutiveErrors} 次无法获取帧，停止分析`);
                                uiController.showAlert(`连续 ${maxConsecutiveErrors} 次无法获取帧，分析已停止`, 'error');
                                this.stopAnalysis();
                                return;
                            }
                        } else {
                            const frameTime = Date.now() / 1000;
                            await this.analyzeFrame(frameData, frameTime);
                            this.frameCount++;
                            updateUI();
                            consecutiveErrors = 0;
                        }
                    } catch (error) {
                        console.error('分析帧时出错:', error);
                        consecutiveErrors++;
                        
                        if (consecutiveErrors >= maxConsecutiveErrors) {
                            console.error(`连续 ${maxConsecutiveErrors} 次分析帧失败，停止分析`);
                            uiController.showAlert(`连续 ${maxConsecutiveErrors} 次分析帧失败，分析已停止`, 'error');
                            this.stopAnalysis();
                            return;
                        }
                    } finally {
                        isProcessing = false;
                    }
                }
                
                // 请求下一帧处理
                if (state.isAnalyzing) {
                    state.streamFrameTimer = requestAnimationFrame(processContinuously);
                }
            };
            
            // 开始连续处理循环
            state.streamFrameTimer = requestAnimationFrame(processContinuously);
            
        } catch (error) {
            console.error('启动视频流分析时出错:', error);
            uiController.showAlert(`启动视频流分析时出错: ${error.message}`, 'error');
            this.stopAnalysis();
        }
    },
    
    /**
     * 开始网络摄像头分析
     */
    async startIpCameraAnalysis() {
        try {
            uiController.showAlert('开始网络摄像头分析', 'info');
            
            // 更新按钮为分析中状态
            this.updateAnalyzeButton('analyzing');
            
            // 记录分析开始时间
            const startTime = Date.now();
            const maxAnalysisTime = 10 * 60 * 1000; // 最大分析时间10分钟
            
            // 立即分析第一帧
            const initialFrame = ipCameraController.captureCurrentFrame();
            if (initialFrame) {
                const initialTime = Date.now() / 1000;
                await this.analyzeFrame(initialFrame, initialTime);
                this.frameCount++;
                this.updateAnalyzeButton('analyzing', this.frameCount);
            } else {
                console.error('无法获取初始帧，请检查网络摄像头连接是否正常');
                uiController.showAlert('无法获取初始帧，请检查网络摄像头连接是否正常', 'error');
                return;
            }
            
            // 确保之前的定时器被清除
            if (state.ipCameraFrameTimer) {
                clearInterval(state.ipCameraFrameTimer);
                state.ipCameraFrameTimer = null;
            }
            
            // 使用更快的防抖时间
            const updateUI = utils.debounce(() => {
                this.updateAnalyzeButton('analyzing', this.frameCount);
            }, 150); // 降低防抖时间，使UI更新更及时
            
            // 创建分析队列，保证一次只处理一帧
            let isProcessing = false;
            let consecutiveErrors = 0;
            const maxConsecutiveErrors = 5; // 最大连续错误次数
            
            // 设置分析间隔（单位：毫秒）
            const analysisInterval = 2000; // 每2秒分析一次
            
            // 开始周期性分析
            state.ipCameraFrameTimer = setInterval(async () => {
                // 检查是否超过最大分析时间
                const currentTime = Date.now();
                const elapsedTime = currentTime - startTime;
                
                if (elapsedTime >= maxAnalysisTime) {
                    uiController.showAlert(`已达到最大分析时间 ${maxAnalysisTime/1000} 秒，自动停止分析`, 'info');
                    this.stopAnalysis();
                    return;
                }
                
                // 检查网络摄像头是否仍然连接
                if (!state.ipCameraActive) {
                    uiController.showAlert('网络摄像头已断开连接，分析已结束', 'info');
                    this.stopAnalysis();
                    return;
                }
                
                if (!state.isAnalyzing) {
                    return;
                }
                
                // 如果当前没有正在处理的帧，获取并处理新帧
                if (!isProcessing) {
                    isProcessing = true;
                    
                    try {
                        // 获取当前帧
                        const frameData = ipCameraController.captureCurrentFrame();
                        if (!frameData) {
                            console.error('无法获取当前帧');
                            consecutiveErrors++;
                            
                            if (consecutiveErrors >= maxConsecutiveErrors) {
                                console.error(`连续 ${maxConsecutiveErrors} 次无法获取帧，停止分析`);
                                uiController.showAlert(`连续 ${maxConsecutiveErrors} 次无法获取帧，分析已停止`, 'error');
                                this.stopAnalysis();
                                return;
                            }
                        } else {
                            const frameTime = Date.now() / 1000;
                            await this.analyzeFrame(frameData, frameTime);
                            this.frameCount++;
                            updateUI();
                            consecutiveErrors = 0;
                        }
                    } catch (error) {
                        console.error('分析帧时出错:', error);
                        consecutiveErrors++;
                        
                        if (consecutiveErrors >= maxConsecutiveErrors) {
                            console.error(`连续 ${maxConsecutiveErrors} 次分析帧失败，停止分析`);
                            uiController.showAlert(`连续 ${maxConsecutiveErrors} 次分析帧失败，分析已停止`, 'error');
                            this.stopAnalysis();
                            return;
                        }
                    } finally {
                        isProcessing = false;
                    }
                }
            }, analysisInterval);
            
        } catch (error) {
            console.error('网络摄像头分析时出错:', error);
            uiController.showAlert(`网络摄像头分析出错: ${error.message}`, 'error');
            this.stopAnalysis();
        }
    },
    
    /**
     * 分析上传的视频
     */
    async analyzeUploadedVideo() {
        try {
            uiController.showAlert(`开始分析 ${state.extractedFrames.length} 帧`, 'info');
            
            // 使用批处理方式分析帧，减少内存占用
            const batchSize = 5; // 每批处理的帧数
            const totalFrames = state.extractedFrames.length;
            
            for (let i = 0; i < totalFrames; i += batchSize) {
                if (!state.isAnalyzing) break;
                
                const batchEnd = Math.min(i + batchSize, totalFrames);
                const batchPromises = [];
                
                for (let j = i; j < batchEnd; j++) {
                    if (!state.isAnalyzing) break;
                    
                    const frameData = state.extractedFrames[j];
                    // 使用用户调整的抽帧间隔计算时间
                    const time = j * state.frameInterval;
                    
                    // 创建分析Promise
                    const analysisPromise = (async () => {
                        let retryCount = 0;
                        let success = false;
                        
                        while (retryCount < this.maxRetries && !success && state.isAnalyzing) {
                            try {
                                await this.analyzeFrame(frameData, time);
                                success = true;
                                this.frameCount++;
                                return true;
                            } catch (error) {
                                retryCount++;
                                console.error(`分析第 ${j + 1} 帧失败,重试第 ${retryCount} 次:`, error);
                                
                                if (retryCount === this.maxRetries) {
                                    console.error(`分析第 ${j + 1} 帧失败,已达到最大重试次数`);
                                    return false;
                                }
                                
                                await new Promise(resolve => setTimeout(resolve, this.retryDelay));
                            }
                        }
                        return false;
                    })();
                    
                    batchPromises.push(analysisPromise);
                }
                
                // 等待当前批次完成
                await Promise.all(batchPromises);
                
                // 更新进度
                const progress = Math.min(((i + batchSize) / totalFrames * 100), 100).toFixed(1);
                this.updateAnalyzeButton('analyzing', this.frameCount);
                uiController.updateProgress(Math.min(i + batchSize, totalFrames), totalFrames);
                
                // 给UI线程一些时间更新
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
            uiController.showAlert('分析完成', 'success');
            this.stopAnalysis();
            
        } catch (error) {
            console.error('分析上传视频时出错:', error);
            throw error;
        }
    },
    
    /**
     * 分析单帧
     * @param {string} frameData - base64编码的图像数据
     * @param {number} time - 时间戳
     */
    async analyzeFrame(frameData, time) {
        let retryCount = 0;
        const maxRetries = 3;
        
        // 使用更短的超时时间，避免长时间等待
        const timeoutDuration = 15000; // 15秒超时，而不是30秒
        
        // 使用Promise.race和AbortController实现更灵活的超时控制
        while (retryCount < maxRetries) {
            // 创建AbortController用于取消请求
            const controller = new AbortController();
            this.activeControllers.add(controller);
            
            try {
                const requestBody = {
                    model: state.selectedModel,
                    image: frameData,
                    targets: state.searchTargets
                };
                
                // 添加时间戳到请求头，帮助服务器优化处理
                const requestHeaders = {
                    'Content-Type': 'application/json',
                    'X-Frame-Timestamp': time.toString(),
                    'X-Analysis-Priority': 'high'
                };
                
                const timeoutId = setTimeout(() => {
                    controller.abort();
                    this.activeControllers.delete(controller);
                }, timeoutDuration);
                
                try {
                    // 创建请求Promise
                    const fetchPromise = fetch('/api/analyze', {
                        method: 'POST',
                        headers: requestHeaders,
                        body: JSON.stringify(requestBody),
                        signal: controller.signal
                    });
                    
                    // 创建超时Promise
                    const timeoutPromise = new Promise((_, reject) => {
                        setTimeout(() => {
                            this.activeControllers.delete(controller);
                            reject(new Error('分析请求超时'));
                        }, timeoutDuration);
                    });
                    
                    // 使用Promise.race在fetch和超时之间竞争
                    const response = await Promise.race([fetchPromise, timeoutPromise]);
                    
                    clearTimeout(timeoutId);
                    this.activeControllers.delete(controller);
                    
                    if (!response.ok) {
                        // 尝试获取错误信息
                        let errorMessage = `分析请求失败: ${response.statusText}`;
                        try {
                            const errorData = await response.json();
                            if (errorData && errorData.message) {
                                errorMessage = errorData.message;
                            }
                        } catch (e) {
                            console.error('解析错误响应失败:', e);
                        }
                        
                        // 特殊处理Minimax API错误
                        if (errorMessage.includes('Minimax API')) {
                            console.error('Minimax API错误，尝试重试...');
                            // 对于Minimax API错误，增加重试延迟
                            await new Promise(resolve => setTimeout(resolve, 2000 * (retryCount + 1)));
                            retryCount++;
                            continue;
                        }
                        
                        throw new Error(errorMessage);
                    }
                    
                    // 使用更快的JSON解析
                    const result = await response.json();
                    
                    // 验证结果格式
                    if (!result || typeof result !== 'object') {
                        console.error('分析结果格式不正确:', result);
                        throw new Error('分析结果格式不正确');
                    }
                    
                    // 确保结果包含必要的字段
                    if (!result.description) {
                        console.warn('分析结果缺少description字段，使用默认值');
                        result.description = '无法获取描述';
                    }
                    
                    if (!result.targets || !Array.isArray(result.targets)) {
                        console.warn('分析结果缺少targets字段或格式不正确，使用默认值');
                        result.targets = state.searchTargets.map(target => ({
                            name: target,
                            found: false
                        }));
                    }
                    
                    const analysisResult = {
                        time: time,
                        description: result.description,
                        targets: result.targets,
                        frameImage: frameData,
                        frameNumber: this.frameCount,
                        timestamp: time
                    };
                    
                    // 使用非阻塞方式添加结果
                    setTimeout(() => {
                        searchResultController.addResult(analysisResult);
                    }, 0);
                    
                    return; // 成功完成，退出重试循环
                    
                } catch (fetchError) {
                    clearTimeout(timeoutId);
                    this.activeControllers.delete(controller);
                    
                    if (fetchError.name === 'AbortError') {
                        throw new Error('分析请求超时');
                    }
                    throw fetchError;
                }
            } catch (error) {
                console.error(`分析帧时出错 (尝试 ${retryCount + 1}/${maxRetries}):`, error);
                
                retryCount++;
                
                if (retryCount >= maxRetries) {
                    console.error(`分析帧失败，已达到最大重试次数 (${maxRetries})`);
                    // 不抛出错误，让分析过程继续
                    return;
                }
                
                // 使用指数退避策略，但最大等待时间较短
                const retryDelay = Math.min(800 * Math.pow(1.5, retryCount - 1), 2000);
                await new Promise(resolve => setTimeout(resolve, retryDelay));
            }
        }
    },
    
    /**
     * 停止分析
     */
    stopAnalysis() {
        if (!state.isAnalyzing) return;
        
        // 更新状态
        state.isAnalyzing = false;
        
        // 取消所有正在进行的API请求
        this.cancelAllRequests();
        
        // 清理定时器
        if (state.streamFrameTimer) {
            cancelAnimationFrame(state.streamFrameTimer);
            state.streamFrameTimer = null;
        }
        
        if (state.ipCameraFrameTimer) {
            clearInterval(state.ipCameraFrameTimer);
            state.ipCameraFrameTimer = null;
        }
        
        // 如果网络摄像头在使用自己的分析方法，停止它
        if (state.ipCameraActive && typeof ipCameraController.stopAnalysis === 'function') {
            try {
                ipCameraController.stopAnalysis();
            } catch (error) {
                console.error('停止网络摄像头分析时出错:', error);
            }
        }
        
        // 更新按钮状态
        this.updateAnalyzeButton('start');
        
        if (this.frameCount > 0) {
            const message = `分析结束，共分析 ${this.frameCount} 帧`;
            uiController.showAlert(message, 'info');
        }
    },
    
    /**
     * 取消所有活动的请求
     */
    cancelAllRequests() {
        // 取消所有活动的AbortController
        for (const controller of this.activeControllers) {
            controller.abort();
        }
        this.activeControllers.clear();
    },
    
    /**
     * 清理资源
     * 在页面卸载或组件销毁时调用
     */
    cleanup() {
        this.stopAnalysis();
        this.frameCount = 0;
        this.cancelAllRequests();
    }
};

export default analysisController; 
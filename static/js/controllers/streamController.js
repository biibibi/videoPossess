import CONFIG from '../config/config.js';
import elements from '../config/elements.js';
import state from '../config/state.js';
import uiController from './uiController.js';

// 视频流控制器
const streamController = {
    /**
     * 启动视频流
     */
    async startStream() {
        try {
            if (state.localCameraActive) return;

            state.isProcessing = true;
            
            // 请求摄像头访问权限
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
            
            state.mediaStream = stream;
            elements.localCameraVideo.srcObject = stream;
            elements.localCameraContainer.classList.add('active');
            
            // 确保视频元素样式正确
            elements.localCameraVideo.style.width = "100%";
            elements.localCameraVideo.style.height = "100%";
            elements.localCameraVideo.style.objectFit = "contain";
            
            elements.localCameraVideo.onloadedmetadata = () => {
                elements.localCameraVideo.play()
                    .then(() => {
                        state.localCameraActive = true;
                        state.isProcessing = false;
                        uiController.showAlert('info', '本地摄像头已连接'); // 更新消息
                    })
                    .catch(error => {
                        console.error('播放摄像头视频失败:', error);
                        this.stopStream();
                        uiController.showAlert('error', '无法启动摄像头');
                    });
            };
            
            elements.localCameraVideo.onerror = (error) => {
                console.error('摄像头视频错误:', error);
                this.stopStream();
                uiController.showAlert('error', '摄像头错误: ' + error);
            };
        } catch (error) {
            console.error('启动本地摄像头失败:', error); // 更新消息
            state.isProcessing = false;
            uiController.showAlert('error', '无法访问摄像头: ' + (error.message || '未授权'));
        }
    },
    
    /**
     * 停止视频流
     */
    stopStream() {
        try {
            if (state.mediaStream) {
                state.mediaStream.getTracks().forEach(track => track.stop());
                state.mediaStream = null;
            }
            
            if (elements.streamVideo.srcObject) {
                elements.streamVideo.srcObject = null;
            }
            
            state.streamActive = false;
            
            if (state.streamFrameTimer) {
                clearInterval(state.streamFrameTimer);
                state.streamFrameTimer = null;
            }
        } catch (error) {
            console.error('停止视频流时出错:', error);
        }
    },
    
    /**
     * 获取当前视频帧
     * @returns {string|null} base64编码的图像数据或null
     */
    getCurrentFrame() {
        try {
            if (!state.streamActive || !elements.streamVideo) {
                return null;
            }
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = elements.streamVideo.videoWidth;
            canvas.height = elements.streamVideo.videoHeight;
            
            ctx.drawImage(elements.streamVideo, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/jpeg', CONFIG.frameQuality);
        } catch (error) {
            console.error('获取当前帧时出错:', error);
            return null;
        }
    },
    
    /**
     * 获取当前抽帧间隔
     * @returns {number} 抽帧间隔（秒）
     */
    getFrameInterval() {
        const frameInterval = elements.autoStreamFrameInterval.checked ? 
               CONFIG.defaultFrameInterval : 
               parseFloat(elements.streamFrameInterval.value);
               
        console.log(`获取抽帧间隔：${frameInterval}秒 (${elements.autoStreamFrameInterval.checked ? '自动模式' : '手动设置'})`);
        return frameInterval;
    }
};

export default streamController; 
 
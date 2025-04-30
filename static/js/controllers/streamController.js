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
            if (state.streamActive) {
                uiController.showAlert('视频流已在运行中', 'info');
                return;
            }

            uiController.showAlert('正在连接摄像头...', 'info');
            
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });
            
            state.mediaStream = stream;
            elements.streamVideo.srcObject = stream;
            
            elements.streamVideo.onloadedmetadata = () => {
                elements.streamVideo.play();
                state.streamActive = true;
                uiController.showAlert('摄像头连接成功', 'success');
            };
            
            elements.streamVideo.onerror = (error) => {
                console.error('视频流加载失败:', error);
                uiController.showAlert('视频流加载失败', 'error');
                this.stopStream();
            };
        } catch (error) {
            console.error('启动视频流时出错:', error);
            uiController.showAlert('无法访问摄像头，请检查权限设置', 'error');
            state.streamActive = false;
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
        return elements.autoStreamFrameInterval.checked ? 
               CONFIG.defaultFrameInterval : 
               parseFloat(elements.streamFrameInterval.value);
    }
};

export default streamController; 
 
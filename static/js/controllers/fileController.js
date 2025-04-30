import elements from '../config/elements.js';
import state from '../config/state.js';
import utils from '../utils/utils.js';
import uiController from './uiController.js';
import analysisController from './analysisController.js';

// 文件控制器
const fileController = {
    async handleFile(file) {
        try {
            if (!file) return;
            
            // 检查文件类型
            if (!file.type.startsWith('video/')) {
                uiController.showAlert('请上传视频文件', 'error');
                return;
            }
            
            // 检查文件大小
            if (file.size > 100 * 1024 * 1024) { // 100MB
                uiController.showAlert('文件大小不能超过100MB', 'error');
                return;
            }
            
            // 预览视频
            uiController.previewVideo(file);
            
            // 初始化抽帧
            await this.initializeFrameExtraction(file);
            
        } catch (error) {
            console.error('处理文件时出错:', error);
            uiController.showAlert('处理文件时出错', 'error');
        }
    },
    
    async initializeFrameExtraction(file) {
        try {
            const video = elements.videoPreview;
            video.src = URL.createObjectURL(file);
            
            await new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    state.videoName = file.name;
                    state.totalFrames = Math.ceil(video.duration / state.frameInterval);
                    state.extractedFrames = [];
                    state.currentFrameIndex = 0;
                    resolve();
                };
            });
            
            uiController.showAlert(`开始抽帧，间隔${state.frameInterval}秒，共${state.totalFrames}帧`);
            await this.extractFrames();
            
        } catch (error) {
            console.error('初始化抽帧时出错:', error);
            uiController.showAlert('初始化抽帧时出错', 'error');
        }
    },
    
    async extractFrames() {
        try {
            const video = elements.videoPreview;
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            for (let i = 0; i < state.totalFrames; i++) {
                video.currentTime = i * state.frameInterval;
                
                await new Promise((resolve) => {
                    video.onseeked = () => {
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const frameData = canvas.toDataURL('image/jpeg', 0.8);
                        state.extractedFrames.push(frameData);
                        state.currentFrameIndex = i + 1;
                        uiController.updateProgress(state.currentFrameIndex, state.totalFrames);
                        resolve();
                    };
                });
            }
            
            uiController.showAlert(`抽帧完成，共${state.extractedFrames.length}帧`);
            analysisController.startAnalysis();
            
        } catch (error) {
            console.error('抽帧时出错:', error);
            uiController.showAlert('抽帧时出错', 'error');
        }
    }
};

export default fileController; 
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
                    // 确保使用用户调整的帧间隔值
                    const frameInterval = parseFloat(elements.frameInterval.value);
                    state.frameInterval = frameInterval;
                    state.totalFrames = Math.ceil(video.duration / frameInterval);
                    state.extractedFrames = [];
                    state.currentFrameIndex = 0;
                    resolve();
                };
            });
            
            uiController.showAlert(`开始抽帧，间隔${state.frameInterval}秒，共${state.totalFrames}帧`);
            await this.extractFrames();

            // Store the video path (assuming file.name is usable by the backend)
            state.uploadedVideoPath = file.name; 
            
            // Show the summarize button
            const videoActions = document.getElementById('videoActions');
            if (videoActions) {
                videoActions.style.display = 'block';
            }
            
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
    },

    async handleSummarizeVideoClick() {
        if (!state.uploadedVideoPath) {
            uiController.showAlert('No video path available for summarization.', 'error');
            console.error('Error: state.uploadedVideoPath is not set.');
            return;
        }

        uiController.showVideoSummaryModal();
        uiController.showSummaryLoading();

        try {
            const response = await fetch('/api/video/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // Sending only video_path, backend will use defaults for other params
                body: JSON.stringify({ video_path: state.uploadedVideoPath }), 
            });

            const data = await response.json();
            uiController.hideSummaryLoading();

            if (response.ok && data.status === 'success') {
                uiController.updateVideoSummaryContent(data.summary, {
                    model_used: data.model_used,
                    frames_processed: data.frames_processed
                });
            } else {
                uiController.showVideoSummaryError(data.message || 'Failed to summarize video.');
                console.error('Summarization API error:', data.message);
            }
        } catch (error) {
            uiController.hideSummaryLoading();
            uiController.showVideoSummaryError('Network error or unexpected issue during summarization.');
            console.error('Error calling summarization API:', error);
        }
    }
};

// Event listener for the summarize button (using event delegation)
document.addEventListener('DOMContentLoaded', () => {
    // Ensure elements are loaded before trying to attach listeners if not using delegation from document
    const dropZone = document.getElementById('dropZone'); // Or any static parent of #videoActions
    if (dropZone) {
        dropZone.addEventListener('click', function(event) {
            if (event.target.closest('.summarize-video-btn')) {
                fileController.handleSummarizeVideoClick();
            }
        });
    } else {
        // Fallback to document if dropZone is not found immediately,
        // though specific delegation is better.
        document.addEventListener('click', function(event) {
            if (event.target.closest('.summarize-video-btn')) {
                fileController.handleSummarizeVideoClick();
            }
        });
    }
});

export default fileController;
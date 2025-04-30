import CONFIG from '../config/config.js';
import elements from '../config/elements.js';
import state from '../config/state.js';
import utils from '../utils/utils.js';

// UI控制器
const uiController = {
    showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        elements.alertContainer.appendChild(alertDiv);
        
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    },
    
    updateProgress(current, total) {
        const progress = (current / total) * 100;
        elements.progressBar.style.width = `${progress}%`;
        elements.progressBar.setAttribute('aria-valuenow', progress);
        elements.progressBar.textContent = `${Math.round(progress)}%`;
    },
    
    resetProgress() {
        elements.progressBar.style.width = '0%';
        elements.progressBar.setAttribute('aria-valuenow', 0);
        elements.progressBar.textContent = '0%';
    },
    
    previewVideo(file) {
        const video = elements.videoPreview;
        video.src = URL.createObjectURL(file);
        video.style.display = 'block';
        elements.uploadArea.style.display = 'none';
    },
    
    cleanupVideoPreview() {
        const video = elements.videoPreview;
        if (video.src) {
            URL.revokeObjectURL(video.src);
        }
        video.src = '';
        video.style.display = 'none';
        elements.uploadArea.style.display = 'block';
    },
    
    resetUploadArea() {
        elements.fileInput.value = '';
        elements.uploadArea.style.display = 'block';
        elements.videoPreview.style.display = 'none';
    },

    updateModelSelection(model) {
        try {
            document.querySelectorAll('.model-card').forEach(card => {
                card.classList.remove('selected');
            });
            document.getElementById(`${model}-model`).classList.add('selected');
            state.selectedModel = model;
        } catch (error) {
            console.error('更新模型选择时出错:', error);
        }
    },

    showVideoPreview(file) {
        try {
            this.cleanupVideoPreview();
            
            state.currentVideoURL = URL.createObjectURL(file);
            elements.videoPreview.src = state.currentVideoURL;
            
            elements.dropZone.classList.add('has-video');
            state.hasUploadedVideo = true;

            elements.videoPreview.onloadeddata = () => {
                elements.videoPreview.play().catch(error => {
                    console.warn('自动播放失败:', error);
                });
            };

            elements.videoPreview.onerror = () => {
                this.showAlert('error', '视频加载失败，请重试');
                this.cleanupVideoPreview();
            };
        } catch (error) {
            console.error('显示视频预览时出错:', error);
            this.showAlert('error', '视频预览失败，请重试');
        }
    },

    updateFrameIntervalDisplay(value, isStream = false) {
        const valueElement = isStream ? elements.streamFrameIntervalValue : elements.frameIntervalValue;
        const slider = isStream ? elements.streamFrameInterval : elements.frameInterval;
        const autoCheck = isStream ? elements.autoStreamFrameInterval : elements.autoFrameInterval;
        
        valueElement.textContent = value;
        slider.value = value;
        slider.disabled = autoCheck.checked;
    }
};

export default uiController; 
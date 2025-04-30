import CONFIG from '../config/config.js';

// 工具函数
const utils = {
    isValidVideoType(type) {
        return CONFIG.validVideoTypes.includes(type);
    },

    isValidFileSize(size) {
        return size <= CONFIG.maxFileSize;
    },

    clearVideoResource() {
        try {
            if (state.currentVideoURL) {
                URL.revokeObjectURL(state.currentVideoURL);
                state.currentVideoURL = null;
            }
        } catch (error) {
            console.error('清理视频资源时出错:', error);
        }
    },

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    },

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    throttle(func, limit) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func(...args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
};

export default utils; 
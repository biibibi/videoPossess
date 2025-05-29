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
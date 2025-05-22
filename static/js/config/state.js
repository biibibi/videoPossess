import CONFIG from './config.js';

// 状态管理
const state = {
    selectedModel: CONFIG.defaultModel,
    hasUploadedVideo: false,
    currentVideoURL: null,
    isProcessing: false,
    streamActive: false,
    mediaStream: null,
    searchTargets: [],
    searchResults: [],
    resultCount: 0,
    videoName: '',
    frameCount: 0,
    extractedFrames: [],
    currentFrameIndex: 0,
    currentMode: 'upload',
    streamFrameInterval: CONFIG.defaultFrameInterval,
    streamFrameTimer: null,
    streamExtractedFrames: [],
    isAnalyzing: false,
    uploadedVideoPath: null, // To store the path of the video for summarization
    
    // 网络摄像头相关状态
    ipCameraActive: false,
    ipCameraUrl: null,
    ipCameraFrameTimer: null,
    manualDisconnect: false, // 标记是否为手动断开连接
    reconnectTimer: null, // 自动重连定时器
    rtcConnection: null // WebRTC连接对象
};

export default state; 
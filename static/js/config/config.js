// 常量配置
const CONFIG = {
    maxFileSize: 100 * 1024 * 1024, // 100MB
    validVideoTypes: ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'],
    alertDuration: 2500,
    defaultModel: 'llama',
    defaultFrameInterval: 0.3,
    minFrameInterval: 0.2,
    maxFrameInterval: 10,
    maxResultsCount: 100,
    maxStreamFrames: 100,
    analysisDelay: 500,
    frameQuality: 0.75
};

export default CONFIG; 
// DOM元素缓存
const elements = {
    init() {
        this.dropZone = document.getElementById('dropZone');
        this.fileInput = document.getElementById('file-input');
        this.videoPreview = document.getElementById('videoPreview');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.streamBtn = document.getElementById('streamBtn');
        this.uploadSection = document.getElementById('uploadSection');
        this.streamSection = document.getElementById('streamSection');
        this.successAlert = document.getElementById('successAlert');
        this.errorAlert = document.getElementById('errorAlert');
        this.errorMessage = document.getElementById('errorMessage');
        this.streamVideo = document.getElementById('streamVideo');
        this.streamContainer = document.querySelector('.stream-container');
        this.searchTargetsContainer = document.querySelector('.search-targets-container');
        this.searchTargetsList = document.getElementById('searchTargetsList');
        this.targetInput = document.getElementById('targetInput');
        this.addTargetBtn = document.getElementById('addTargetBtn');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.searchResultsList = document.getElementById('searchResultsList');
        this.noResultsMessage = document.getElementById('noResultsMessage');
        this.resultCount = document.getElementById('resultCount');
        this.frameInterval = document.getElementById('frameInterval');
        this.frameIntervalValue = document.getElementById('frameIntervalValue');
        this.autoFrameInterval = document.getElementById('autoFrameInterval');
        this.streamFrameInterval = document.getElementById('streamFrameInterval');
        this.streamFrameIntervalValue = document.getElementById('streamFrameIntervalValue');
        this.autoStreamFrameInterval = document.getElementById('autoStreamFrameInterval');
    }
};

export default elements; 
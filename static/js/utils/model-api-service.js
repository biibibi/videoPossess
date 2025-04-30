/**
 * 模型API服务
 * 负责处理与各种视觉语言模型的API交互
 */
const ModelAPIService = {
    /**
     * 分析图像
     * @param {string} modelName - 模型名称
     * @param {string} frameImage - base64编码的图像
     * @param {Array<string>} searchTargets - 搜索目标列表
     * @returns {Promise<Object>} - 分析结果
     */
    async analyzeImage(modelName, frameImage, searchTargets) {
        try {
            // 验证输入参数
            if (!modelName || !frameImage || !searchTargets || !Array.isArray(searchTargets)) {
                throw new Error('无效的输入参数');
            }

            // 构建请求数据
            const requestData = {
                model: modelName,
                image: frameImage,
                targets: searchTargets
            };

            // 重试相关配置
            const maxRetries = 3;
            const baseDelayMs = 2000; // 2秒基础延迟
            let retryCount = 0;

            while (retryCount <= maxRetries) {
                try {
                    // 添加时间戳和优先级信息
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-Analysis-Priority': 'normal',
                            'X-Frame-Timestamp': Date.now().toString()
                        },
                        body: JSON.stringify(requestData)
                    });

                    // 处理速率限制错误 (429)
                    if (response.status === 429) {
                        if (retryCount < maxRetries) {
                            // 使用指数退避策略计算等待时间
                            const delayMs = baseDelayMs * Math.pow(2, retryCount);
                            console.warn(`API速率限制 (429)，将在 ${delayMs/1000} 秒后重试 (${retryCount+1}/${maxRetries})`);
                            
                            if (window.uiController) {
                                window.uiController.showAlert('error', `请求频率过高，${delayMs/1000}秒后重试...`);
                            }
                            
                            // 等待后重试
                            await new Promise(resolve => setTimeout(resolve, delayMs));
                            retryCount++;
                            continue;
                        } else {
                            // 达到最大重试次数
                            const errorData = await response.json().catch(() => ({}));
                            const errorMsg = errorData.message || '请求频率过高，请稍后再试';
                            console.error(`达到最大重试次数 (${maxRetries})，API速率限制错误:`, errorMsg);
                            
                            // 返回一个降级的结果
                            return {
                                description: '由于请求频率限制，无法完成分析。请稍后再试。',
                                targets: searchTargets.map(target => ({
                                    name: target,
                                    found: false
                                }))
                            };
                        }
                    }

                    // 其他错误
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({}));
                        const errorMsg = errorData.message || `服务器返回错误: ${response.status}`;
                        
                        if (retryCount < maxRetries && [500, 502, 503, 504].includes(response.status)) {
                            // 服务器错误，可以重试
                            const delayMs = baseDelayMs * Math.pow(2, retryCount);
                            console.warn(`服务器错误 (${response.status})，将在 ${delayMs/1000} 秒后重试 (${retryCount+1}/${maxRetries})`);
                            
                            // 等待后重试
                            await new Promise(resolve => setTimeout(resolve, delayMs));
                            retryCount++;
                            continue;
                        }
                        
                        throw new Error(errorMsg);
                    }

                    // 成功响应 - 解析JSON
                    const result = await response.json().catch(e => {
                        console.error("JSON解析错误:", e);
                        throw new Error("响应格式无效，无法解析JSON");
                    });
                    
                    // 记录原始API响应，便于调试
                    console.log(`${modelName}模型API原始响应:`, result);
                    
                    // 确保结果包含预期的字段，并正确保留每个目标的found状态
                    const processedResult = {
                        description: result.description || '无描述',
                        targets: this._processTargetsArray(result.targets, searchTargets)
                    };
                    
                    // 记录处理后的结果，便于调试
                    console.log(`${modelName}模型处理后的结果:`, processedResult);
                    
                    return processedResult;
                } catch (fetchError) {
                    // 处理网络相关错误，这些错误可以重试
                    if (retryCount < maxRetries && 
                        (fetchError.name === 'TypeError' || fetchError.message.includes('network') || fetchError.message.includes('timeout'))) {
                        const delayMs = baseDelayMs * Math.pow(2, retryCount);
                        console.warn(`网络错误，将在 ${delayMs/1000} 秒后重试 (${retryCount+1}/${maxRetries}): ${fetchError.message}`);
                        
                        // 等待后重试
                        await new Promise(resolve => setTimeout(resolve, delayMs));
                        retryCount++;
                        continue;
                    }
                    
                    // 其他错误直接抛出
                    throw fetchError;
                }
            }
            
            // 如果达到这里，说明重试次数已用完但仍未成功
            throw new Error('多次请求失败，请稍后再试');
        } catch (error) {
            console.error('分析图像时出错:', error);
            
            // 向UI显示错误信息
            if (window.uiController) {
                window.uiController.showAlert('error', error.message || '分析图像失败');
            }
            
            // 返回降级结果
            return {
                description: '分析失败: ' + error.message,
                targets: searchTargets.map(target => ({
                    name: target,
                    found: false
                }))
            };
        }
    },
    
    /**
     * 处理目标数组，确保每个目标都有正确的found属性
     * @private
     * @param {Array|null} targetsFromAPI - API返回的目标数组
     * @param {Array<string>} originalTargets - 原始搜索目标列表
     * @returns {Array} - 处理后的目标数组
     */
    _processTargetsArray(targetsFromAPI, originalTargets) {
        // 如果API没有返回有效的目标数组，使用原始目标创建一个默认的
        if (!targetsFromAPI || !Array.isArray(targetsFromAPI) || targetsFromAPI.length === 0) {
            console.warn("API返回的targets无效，使用默认目标");
            return originalTargets.map(target => ({
                name: target,
                found: false
            }));
        }
        
        // 确保每个目标都有name和found属性
        return targetsFromAPI.map(target => {
            // 检查target是否是有效的对象
            if (!target || typeof target !== 'object') {
                console.warn("目标格式无效:", target);
                return { name: "未知目标", found: false };
            }
            
            // 确保target.name是有效的字符串
            const name = typeof target.name === 'string' && target.name ? 
                          target.name : 
                          (typeof target === 'string' ? target : "未知目标");
            
            // 确保target.found是布尔值
            const found = typeof target.found === 'boolean' ? 
                          target.found : 
                          false;
            
            return { name, found };
        });
    }
};

// 导出服务
window.ModelAPIService = ModelAPIService; 
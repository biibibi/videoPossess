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
            const baseDelayMs = 2000;
            let retryCount = 0;

            while (retryCount < maxRetries) {
                try {
                    // 发送请求
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-Analysis-Priority': 'high',
                            'X-Frame-Timestamp': new Date().toISOString()
                        },
                        body: JSON.stringify(requestData)
                    });

                    // 检查响应状态
                    if (!response.ok) {
                        const errorText = await response.text();
                        console.error(`API请求失败: ${response.status} ${response.statusText}`, errorText);
                        
                        // 根据状态码确定是否需要重试
                        if (response.status >= 500 && retryCount < maxRetries - 1) {
                            retryCount++;
                            const delay = baseDelayMs * (2 ** retryCount) * (0.5 + Math.random() * 0.5);
                            // 服务器错误重试
                            await new Promise(resolve => setTimeout(resolve, delay));
                            continue;
                        }
                        
                        // 返回错误结果
                        return {
                            description: `API请求失败: ${response.status} ${response.statusText}`,
                            targets: searchTargets.map(target => ({
                                name: target,
                                found: false
                            }))
                        };
                    }

                    // 使用增强的响应处理函数
                    return await this._processResponse(response, searchTargets, modelName);
                    
                } catch (fetchError) {
                    console.error(`API请求过程中出错:`, fetchError);
                    
                    // 网络错误通常值得重试
                    if (retryCount < maxRetries - 1) {
                        retryCount++;
                        const delay = baseDelayMs * (2 ** retryCount) * (0.5 + Math.random() * 0.5);
                        // 网络错误重试
                        await new Promise(resolve => setTimeout(resolve, delay));
                        continue;
                    }
                    
                    // 最后一次尝试也失败了，返回错误结果
                    return {
                        description: `请求失败: ${fetchError.message || '网络错误'}`,
                        targets: searchTargets.map(target => ({
                            name: target,
                            found: false
                        }))
                    };
                }
            }
            
            // 不应该到达这里，但以防万一
            throw new Error('已超过最大重试次数');
            
        } catch (error) {
            console.error('分析图像时出错:', error);
            
            // 返回错误结果
            return {
                description: `处理错误: ${error.message || '未知错误'}`,
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
     * @param {string} modelName - 模型名称，用于日志
     * @returns {Array} - 处理后的目标数组
     */
    _processTargetsArray(targetsFromAPI, originalTargets, modelName = "未知模型") {
        // 如果API没有返回有效的目标数组，使用原始目标创建一个默认的
        if (!targetsFromAPI || !Array.isArray(targetsFromAPI) || targetsFromAPI.length === 0) {
            console.warn(`[${modelName}] API返回的targets无效，使用默认目标`);
            return originalTargets.map(target => ({
                name: target,
                found: false
            }));
        }
        
        // 特殊处理：检测targetsFromAPI是否为字符串(非标准格式)
        if (typeof targetsFromAPI === 'string') {
            console.warn(`[${modelName}] API返回的targets是字符串而非数组，尝试解析:`, targetsFromAPI);
            try {
                // 处理Python字典格式的字符串
                if (targetsFromAPI.includes("'") && !targetsFromAPI.includes('"')) {
                    // 将Python风格的字符串转为JSON风格
                    const jsonString = targetsFromAPI
                        .replace(/'/g, '"')           // 单引号替换为双引号
                        .replace(/True/g, 'true')     // Python True替换为JSON true
                        .replace(/False/g, 'false');  // Python False替换为JSON false
                    
                    // 尝试解析转换后的字符串
                    try {
                        const parsedTargets = JSON.parse(jsonString);
                        if (Array.isArray(parsedTargets)) {
                            targetsFromAPI = parsedTargets;
                        }
                    } catch (parseError) {
                        console.error(`[${modelName}] 解析转换后的字符串失败:`, parseError);
                    }
                }
            } catch (error) {
                console.error(`[${modelName}] 处理字符串格式的targets时出错:`, error);
            }
            
            // 如果处理后仍然不是数组，返回默认值
            if (!Array.isArray(targetsFromAPI)) {
                return originalTargets.map(target => ({
                    name: target,
                    found: false
                }));
            }
        }
        
        // 创建一个映射，用于查找目标名称对应的API返回的found状态
        const targetMap = new Map();
        
        // 将API返回的目标数据转换为映射
        for (const target of targetsFromAPI) {
            // 跳过无效的目标
            if (!target || typeof target !== 'object') {
                continue;
            }
            
            // 确保target.name是有效的字符串
            const name = typeof target.name === 'string' && target.name ? 
                          target.name : 
                          (typeof target === 'string' ? target : "未知目标");
            
            // 确保target.found是布尔值，如果不是，记录警告
            if (typeof target.found !== 'boolean') {
                console.warn(`[${modelName}] 目标 "${name}" 的found属性不是布尔值:`, target.found);
            }
            
            // 明确转换为布尔值
            const found = target.found === true;
            
            // 保存到映射
            targetMap.set(name.toLowerCase(), found);
        }
        
        // 确保每个原始目标都有正确的found状态
        const processedTargets = originalTargets.map(targetName => {
            const normalizedName = typeof targetName === 'string' ? targetName.toLowerCase() : String(targetName).toLowerCase();
            // 查找这个目标在API返回中的found状态
            const found = targetMap.has(normalizedName) ? targetMap.get(normalizedName) : false;
            
            return {
                name: targetName,
                found: found
            };
        });
        
        return processedTargets;
    },

    async _processResponse(response, searchTargets, modelName) {
        try {
            // 处理响应格式
            let responseData;
            
            // 尝试解析JSON
            try {
                responseData = await response.json();
                
                // 特殊处理：检测是否是Gemini响应
                if (modelName === "gemini") {
                    
                    // 检查API响应中是否存在额外的JSON字符串
                    if (responseData.description && typeof responseData.description === 'string') {
                        const description = responseData.description;
                        
                        // 检查描述中是否嵌入了JSON
                        const jsonMatch = description.match(/```json\s*([\s\S]*?)\s*```/) || 
                                         description.match(/```\s*([\s\S]*?)\s*```/) ||
                                         description.match(/\{[\s\S]*"targets"[\s\S]*\}/);
                                         
                        if (jsonMatch) {
                            try {
                                // 提取并解析嵌入的JSON
                                const embeddedJson = jsonMatch[1] || jsonMatch[0];
                                const cleanedJson = embeddedJson
                                    .replace(/'/g, '"')
                                    .replace(/True/g, 'true')
                                    .replace(/False/g, 'false');
                                
                                const parsedData = JSON.parse(cleanedJson);
                                console.log(`[Gemini] 成功从描述中提取JSON:`, parsedData);
                                
                                // 合并提取的JSON内容
                                if (parsedData.targets && Array.isArray(parsedData.targets)) {
                                    responseData.targets = parsedData.targets;
                                }
                                if (parsedData.description) {
                                    responseData.description = parsedData.description;
                                }
                            } catch (error) {
                                console.warn(`[Gemini] 提取JSON失败:`, error);
                            }
                        }
                    }
                }
            } catch (jsonError) {
                // JSON解析失败，尝试获取文本内容
                console.warn(`[${modelName}] JSON解析失败:`, jsonError);
                const text = await response.text();
                console.log(`[${modelName}] 获取到文本响应:`, text);
                
                // 尝试处理可能是Python风格的字典字符串
                try {
                    // 检查是否是Python字典格式 (使用单引号和True/False)
                    if (text.includes("'") && (text.includes("{") || text.includes("["))) {
                        console.log(`[${modelName}] 检测到可能是Python字典格式，尝试转换`);
                        
                        // 清理前缀，例如 ```json 前缀
                        let cleanText = text.replace(/```json\s*/g, '').replace(/```\s*$/g, '');
                        
                        // 转换Python风格为JSON风格
                        cleanText = cleanText
                            .replace(/'/g, '"')           // 单引号替换为双引号
                            .replace(/True/g, 'true')     // Python True替换为JSON true
                            .replace(/False/g, 'false');  // Python False替换为JSON false
                        
                        // 尝试解析转换后的文本
                        try {
                            responseData = JSON.parse(cleanText);
                            console.log(`[${modelName}] 成功将Python风格字典转换为JSON:`, responseData);
                        } catch (parseError) {
                            console.error(`[${modelName}] 转换后解析失败:`, parseError);
                            // 创建默认响应
                            responseData = {
                                description: text,
                                status: "error",
                                targets: [],
                                message: "无法解析响应格式"
                            };
                        }
                    } else {
                        // 不是Python字典格式，创建默认响应
                        responseData = {
                            description: text,
                            status: "success",
                            targets: []
                        };
                    }
                } catch (textProcessError) {
                    console.error(`[${modelName}] 处理文本响应时出错:`, textProcessError);
                    responseData = {
                        description: text,
                        status: "error",
                        targets: [],
                        message: "处理响应时出错"
                    };
                }
            }
            
            // 提取描述
            let description = "";
            if (responseData.description) {
                description = responseData.description;
            } else if (responseData.response) {
                description = responseData.response;
            } else if (responseData.message) {
                description = responseData.message;
            } else if (responseData.error) {
                description = responseData.error;
            } else if (typeof responseData === "string") {
                description = responseData;
            }
            
            console.log(`[${modelName}] 提取的描述:`, description);
            
            // 处理目标
            let targets = [];
            
            // 如果API返回了targets数组，处理它
            if (responseData.targets && Array.isArray(responseData.targets)) {
                // 使用专用函数处理targets数组
                targets = this._processTargetsArray(responseData.targets, searchTargets, modelName);
            } else if (typeof responseData.targets === 'string') {
                // 如果targets是字符串，尝试解析它
                console.warn(`[${modelName}] targets是字符串而非数组:`, responseData.targets);
                try {
                    // 解析一个可能是字符串格式的JSON
                    const parsedTargets = JSON.parse(responseData.targets
                                                  .replace(/'/g, '"')
                                                  .replace(/True/g, 'true')
                                                  .replace(/False/g, 'false'));
                    
                    if (Array.isArray(parsedTargets)) {
                        targets = this._processTargetsArray(parsedTargets, searchTargets, modelName);
                    } else {
                        targets = searchTargets.map(target => ({
                            name: target,
                            found: false
                        }));
                    }
                } catch (parseError) {
                    console.error(`[${modelName}] 解析targets字符串时出错:`, parseError);
                    targets = searchTargets.map(target => ({
                        name: target,
                        found: false
                    }));
                }
            } else {
                // 如果API没有返回targets，从描述推断targets状态
                console.warn(`[${modelName}] API响应中没有targets数组，从描述推断`);
                const descriptionLower = description.toLowerCase();
                
                targets = searchTargets.map(target => {
                    const targetLower = target.toLowerCase();
                    
                    // 确定目标是否找到 - 简单启发式算法
                    let found = false;
                    
                    // 如果描述中明确包含了目标名称
                    if (descriptionLower.includes(targetLower)) {
                        // 默认假设找到了，除非有明确的否定短语
                        found = true;
                        
                        // 检查否定短语
                        const negationPhrases = [
                            `没有${targetLower}`, `未找到${targetLower}`,
                            `未发现${targetLower}`, `不包含${targetLower}`,
                            `no ${targetLower}`, `not found ${targetLower}`, 
                            `cannot see ${targetLower}`, `doesn't contain ${targetLower}`
                        ];
                        
                        // 如果有任何否定短语，则认为未找到
                        if (negationPhrases.some(phrase => descriptionLower.includes(phrase))) {
                            found = false;
                        }
                    }
                    
                    return { name: target, found };
                });
            }
            
            console.log(`[${modelName}] 最终处理结果: 描述=${description.substring(0, 50)}..., 目标数=${targets.length}`);
            
            return {
                description,
                targets
            };
        } catch (error) {
            console.error(`[${modelName}] 处理API响应时出错:`, error);
            return {
                description: `处理API响应时出错: ${error.message || '未知错误'}`,
                targets: searchTargets.map(target => ({
                    name: target,
                    found: false
                }))
            };
        }
    }
};

// 导出服务
window.ModelAPIService = ModelAPIService; 
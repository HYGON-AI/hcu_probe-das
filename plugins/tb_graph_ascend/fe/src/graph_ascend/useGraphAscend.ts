/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

import request from '../utils/request';
import { LoadGraphFileInfoListType, SelectionType } from './type';
const useGraphAscend = () => {

    const loadGraphFileInfoList = async (): Promise<LoadGraphFileInfoListType> => {
        try {

            const result = await request({ url: 'load_meta_dir', method: 'GET' });
            return result as unknown as LoadGraphFileInfoListType;
        } catch (err) {
            return {
                data: {},
                error: [
                    {
                        run: '',
                        tag: '',
                        info: '加载文件列表失败',
                    }
                ],
            };
        }
    };
    const loadGraphConfig = async (metaData: SelectionType): Promise<any> => {
        const result = await request({ url: 'loadGraphConfigInfo', method: 'POST', data: { metaData } }); // 获取异步的 ArrayBuffer
        return result;
    };

    const loadGraphAllNodeList = async (metaData: SelectionType): Promise<any> => {
        const result = await request({ url: 'loadGraphAllNodeList', method: 'POST', data: { metaData } }); // 获取异步的 ArrayBuffer
        return result;
    };

    return {
        loadGraphConfig,
        loadGraphAllNodeList,
        loadGraphFileInfoList,
    };
};
export default useGraphAscend;

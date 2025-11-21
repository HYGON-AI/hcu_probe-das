/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

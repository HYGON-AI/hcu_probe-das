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

import { isEmpty } from 'lodash';
import request from '../../../utils/request';
import { UseMatchedType, MatchResultType } from '../../type';
const useMatched = (): UseMatchedType => {
  const requestAddMatchNodes = async (npuNodeName: string, benchNodeName: string, metaData: any, isMatchChildren: boolean): Promise<any> => {
    const params = {
      npuNodeName: npuNodeName,
      benchNodeName: benchNodeName,
      metaData,
      isMatchChildren
    };
    const mactchResult = await request({ url: 'addMatchNodes', method: 'POST', data: params });
    return mactchResult;
  };

  const requestDeleteMatchNodes = async (npuNodeName: string, benchNodeName: string, metaData: any, isUnMatchChildren: boolean): Promise<any> => {
    const params = {
      npuNodeName: npuNodeName,
      benchNodeName: benchNodeName,
      metaData,
      isUnMatchChildren
    };
    const mactchResult = await request({ url: 'deleteMatchNodes', method: 'POST', data: params });
    return mactchResult;
  };

  const saveMatchedNodesLink = async (selection: any): Promise<any> => {
    const saveResult = await request({ url: 'saveData', method: 'POST', data: { metaData: selection } });
    return saveResult;
  };
  const saveMatchedRelations = async (selection: any): Promise<any> => {
    const saveResult = await request({ url: 'saveMatchedRelations', method: 'POST', data: { metaData: selection } });
    return saveResult;
  };

  const addMatchedNodesLinkByConfigFile = async (configFile: string, selection: any): Promise<MatchResultType> => {
    if (isEmpty(configFile)) {
      return {
        success: false,
        error: '请选择配置文件',
      };
    }
    const params = {
      configFile: configFile,
      metaData: selection,
    };
    const mactchResult = await request({ url: 'addMatchNodesByConfig', method: 'POST', data: params });

    return mactchResult as MatchResultType;
  };

  const addMatchedNodesLink = async (
    npuNodeName: string,
    benchNodeName: string,
    selection: any,
    isMatchChildren: boolean
  ): Promise<MatchResultType> => {
    if (isEmpty(npuNodeName) || isEmpty(benchNodeName)) {
      return {
        success: false,
        error: '调试侧节点或标杆节点为空',
      };
    }
    const matchResult: MatchResultType = await requestAddMatchNodes(npuNodeName, benchNodeName, selection, isMatchChildren);
    return matchResult;
  };

  const deleteMatchedNodesLink = async (npuNodeName: string, benchNodeName: string, selection: any, isUnMatchChildren: boolean): Promise<any> => {
    if (isEmpty(npuNodeName) || isEmpty(benchNodeName)) {
      return {
        success: false,
        error: '调试侧节点或标杆节点为空',
      };
    }
    const matchResult: MatchResultType = await requestDeleteMatchNodes(npuNodeName, benchNodeName, selection, isUnMatchChildren);
    return matchResult;
  };

  return {
    saveMatchedNodesLink,
    addMatchedNodesLink,
    saveMatchedRelations,
    deleteMatchedNodesLink,
    addMatchedNodesLinkByConfigFile,
  };
};

export default useMatched;

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

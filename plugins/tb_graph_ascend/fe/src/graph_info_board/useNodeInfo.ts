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
import { isEmpty, cloneDeep } from 'lodash';
import useNodeInfoDomain from './domain/useNodeInfoDomain';
import type { NodeInfoResult, NodeInfoType } from './type';
import { BENCH_PREFIX, NPU_PREFIX } from '../common/constant';
import { safeJSONParse } from '../utils';

export interface UseNodeInfoType {
  getNodeInfo: (
    nodeInfo: {
      nodeName: string;
      nodeType: string;
    },
    metaData: any,
  ) => Promise<NodeInfoResult>;
  getIoDataSet: (
    npuNode: any,
    benchNode: any,
    type: 'inputData' | 'outputData',
  ) => {
    matchedIoDataset: Array<Record<string, unknown>>;
    unMatchedNpuIoDataset: Array<Record<string, unknown>>;
    unMatchedBenchIoDataset: Array<Record<string, unknown>>;
  };
  getDetailDataSet: (npuNode: any, benchNode: any) => Array<Record<string, unknown>>;
}

const useNodeInfo = (): UseNodeInfoType => {
  const useNodeInfoService = useNodeInfoDomain();
  /**
   * 获取节点信息
   * @param node_name
   * @param graph_name
   * @param run_name
   * @returns
   */
  const getNodeInfo = async (
    nodeInfo: { nodeName: string; nodeType: string },
    metaData: any,
  ): Promise<NodeInfoResult> => {
    const mactchResult = await useNodeInfoService.getMatchNodeInfo(nodeInfo, metaData);
    if (mactchResult.success && mactchResult.data) {
      mactchResult.data.npu = convertNodeInfo(mactchResult.data?.npu); // 提取有效数据，统一命名
      mactchResult.data.bench = convertNodeInfo(mactchResult.data?.bench); // 提取有效数据，统一命名
    }
    return mactchResult;
  };

  const convertNodeInfo = (nodeInfo: any): NodeInfoType => {
    if (isEmpty(nodeInfo)) {
      return {};
    }
    return {
      name: nodeInfo.id,
      inputData: nodeInfo.input_data,
      outputData: nodeInfo.output_data,
      stackData: !isEmpty(nodeInfo.stack_info) ? JSON.stringify(nodeInfo.stack_info) : '',
      suggestions: nodeInfo.suggestions,
      parallelMergeInfo: !isEmpty(nodeInfo.parallel_merge_info) ? JSON.stringify(nodeInfo.parallel_merge_info) : '',
    };
  };

  /**
   * 获取匹配的输入输出数据
   * @param npuNode NPU节点信息
   * @param benchNode 匹配的节点信息
   * @param name 'inputData' | 'outputData'
   * @returns { matchedDataset: Array<{}>, unMatchedNpuDataset: Array<{}>, unMatchedBenchDataset: Array<{}> }
   */
  const getIoDataSet = (
    npuNode: any,
    benchNode: any,
    type: 'inputData' | 'outputData',
  ): {
    matchedIoDataset: Array<Record<string, unknown>>;
    unMatchedNpuIoDataset: Array<Record<string, unknown>>;
    unMatchedBenchIoDataset: Array<Record<string, unknown>>;
  } => {
    if (isEmpty(npuNode?.[type]) && isEmpty(benchNode?.[type])) {
      return {
        matchedIoDataset: [],
        unMatchedNpuIoDataset: [],
        unMatchedBenchIoDataset: [],
      };
    }
    const npuNodeName = npuNode?.name?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
    const benchNodeName = benchNode?.name?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
    const npuData = cloneDeep(npuNode?.[type]); // 获取当前节点的输入数据
    const benchData = cloneDeep(benchNode?.[type]); // 获取匹配节点的输入数据

    const matchedIoDataset: Array<Record<string, unknown>> = []; // 初始化输入数据集
    const unMatchedBenchIoDataset: Array<Record<string, unknown>> = [];
    const unMatchedNpuIoDataset: Array<Record<string, unknown>> = [];
    const npuKeys = Object.keys(npuData || {});
    const benchKeys = Object.keys(benchData || {});
    const minLength = Math.min(npuKeys.length, benchKeys.length);
    for (let i = 0; i < minLength; i++) {
      const npuKey = npuKeys[i];
      const benchKey = benchKeys[i];
      matchedIoDataset.push({
        name: npuKey.replace(`${npuNodeName}.`, ''),
        isMatched: true,
        ...npuData[npuKey],
      });
      matchedIoDataset.push({
        name: benchKey.replace(`${benchNodeName}.`, ''),
        isBench: true,
        isMatched: true,
        ...benchData[benchKey],
      });
      delete npuData[npuKey];
      delete benchData[benchKey];
    }
    Object.keys(npuData || {}).forEach((key) => {
      if (npuData[key] !== 'None') {
        unMatchedNpuIoDataset.push({
          name: key.replace(`${npuNodeName}.`, ''),
          ...npuData[key],
        });
      }
    });
    Object.keys(benchData || {}).forEach((key) => {
      if (benchData[key] !== 'None') {
        unMatchedBenchIoDataset.push({
          name: key.replace(`${benchNodeName}.`, ''),
          isBench: true,
          ...benchData[key],
        });
      }
    });
    return { matchedIoDataset, unMatchedNpuIoDataset, unMatchedBenchIoDataset };
  };

  const getDetailDataSet = (npuNode: any, benchNode: any): Array<Record<string, unknown>> => {
    if (isEmpty(npuNode) && isEmpty(benchNode)) {
      return [];
    }
    const nodeName = `NPU：${npuNode?.name?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '')}`;
    const benchNodeName = `Debug：${benchNode?.name?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '')}`;
    const detailData: Array<Record<string, unknown>> = [];
    // 获取stackInfo
    const stackInfo: Record<string, unknown> = {};
    const npustackInfo = npuNode?.stackData;
    const benchstackInfo = benchNode?.stackData;
    const title = 'title';
    if (!isEmpty(npustackInfo)) {
      stackInfo[nodeName] = safeJSONParse(npustackInfo)?.join('\n');
    }
    if (!isEmpty(benchstackInfo)) {
      stackInfo[benchNodeName] = safeJSONParse(benchstackInfo).join('\n');
    }
    if (!isEmpty(stackInfo)) {
      stackInfo[title] = 'stackInfo';
      detailData.push(stackInfo);
    }
    // 获取parallel_merge_info
    const parallelMergeInfo: Record<string, unknown> = {};
    const npuparallelMergeInfo = npuNode?.parallelMergeInfo;
    const benchparallelMergeInfo = benchNode?.parallelMergeInfo;
    if (!isEmpty(npuparallelMergeInfo)) {
      parallelMergeInfo[nodeName] = safeJSONParse(npuparallelMergeInfo).join('\n');
    }
    if (!isEmpty(benchparallelMergeInfo)) {
      parallelMergeInfo[benchNodeName] = safeJSONParse(benchparallelMergeInfo).join('\n');
    }
    if (!isEmpty(parallelMergeInfo)) {
      parallelMergeInfo[title] = 'parallelMergeInfo';
      detailData.push(parallelMergeInfo);
    }
    return detailData;
  };

  return {
    getNodeInfo,
    getIoDataSet,
    getDetailDataSet,
  };
};

export default useNodeInfo;

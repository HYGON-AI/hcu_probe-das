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
export interface MinimapVis {
  npu: boolean;
  bench: boolean;
}
export type Dataset = Array<RunItem>;

export type MetaDirType = {
  [string]: {
    type: string;
    tags: string[];
  }
};

export interface UseMatchedType {
  saveMatchedNodesLink: (selection: any) => Promise<any>;
  addMatchedNodesLink: (npuNodeName: string, benchNodeName: string, selection: any, isMatchChildren: boolean) => Promise<MatchResultType>;
  deleteMatchedNodesLink: (npuNodeName: string, benchNodeName: string, selection: any, isUnMatchChildren: boolean) => Promise<MatchResultType>;
  saveMatchedRelations: (selection: any) => Promise<any>;
  addMatchedNodesLinkByConfigFile: (condfigFile: string, selection: any) => Promise<MatchResultType>;
}
export interface MatchResultType {
  success: boolean;
  error: string;
  data?: {
    npuMatchNodes: Record<string, string>;
    benchMatchNodes: Record<string, string>;
    npuUnMatchNodes: string[];
    benchUnMatchNodes: string[];
    matchResult?: boolean[];
  };
};

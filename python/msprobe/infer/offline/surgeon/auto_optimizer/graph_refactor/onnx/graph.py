# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import itertools
import os
import tempfile
import warnings
from collections import deque
from typing import List, Dict, Union, Sequence, Optional, Tuple, Set

import numpy as np
import onnx

from msprobe.infer.offline.surgeon.auto_optimizer.common.utils import check_output_model_path
from msprobe.infer.offline.surgeon.auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from msprobe.infer.offline.surgeon.auto_optimizer.graph_refactor.interface.base_node import Initializer, PlaceHolder, \
    Node
from msprobe.infer.offline.surgeon.auto_optimizer.graph_refactor.onnx.node import OnnxPlaceHolder, OnnxInitializer, \
    OnnxNode
from msprobe.infer.offline.common import logger
from msprobe.infer.utils.check.rule import Rule
from msprobe.infer.utils.constants import ONNX_MODEL_MAX_SIZE
from msprobe.infer.utils.file_utils import check_input_file_path
from onnx import helper, GraphProto, ModelProto, OperatorSetIdProto, version_converter


class OnnxGraph(BaseGraph):

    def __init__(
            self,
            name: str,
            nodes: Optional[List[OnnxNode]] = None,
            inputs: Optional[List[OnnxPlaceHolder]] = None,
            outputs: Optional[List[OnnxPlaceHolder]] = None,
            initializers: Optional[List[OnnxInitializer]] = None,
            value_infos: Optional[List[OnnxPlaceHolder]] = None,
            **kwargs: Dict[str, object],
    ):
        super(OnnxGraph, self).__init__(name, nodes, inputs, outputs, initializers, value_infos)

        opsets = kwargs.get('opset_imports', 11)
        if isinstance(opsets, int):
            opset_imports = onnx.OperatorSetIdProto()
            opset_imports.version = opsets
            opset_imports = [opset_imports]
        elif isinstance(opsets, Sequence):
            opset_imports = [op for op in opsets if not op.domain or op.domain == '']
            if len(opset_imports) < len(opsets):
                warnings.warn('Only one domain version is allowed, keep opset with domain "ai.onnx"')
        else:
            opset_imports = opsets

        self._meta = {
            'ir_version': kwargs.get('ir_version', 4),
            'producer_name': kwargs.get('producer_name', 'AutoOptimizer'),
            'producer_version': kwargs.get('producer_version', 'alpha'),
            'domain': kwargs.get('domain', ''),
            'model_version': kwargs.get('model_version', 0),
            'opset_imports': opset_imports,
        }

        self._value_infos = []
        self._value_map = {}

    @property
    def opset_imports(self) -> Optional[Sequence[OperatorSetIdProto]]:
        return self._meta.get('opset_imports')

    @opset_imports.setter
    def opset_imports(self, opset: Union[int, None]) -> None:
        if not opset:
            self._meta['opset_imports'] = None
        else:
            opset_imports = OperatorSetIdProto()
            opset_imports.version = opset
            model = self.model()
            converted_model = version_converter.convert_version(model, opset)
            self.graph = OnnxGraph.parse(converted_model)
            self._meta['opset_imports'] = [opset_imports]

    @staticmethod
    def connect_graph(graph1: 'OnnxGraph', graph2: 'OnnxGraph', io_map: List[str], graph_name: str):
        """Implementation of concatenating two graphs based on the io_map.

        1. Connect the outputs of the first graph and inputs of the second one.
        2. Update value_infos, initializers, nodes, node_map, prev_map, next_map

        Arguments:
            graph1 (OnnxGraph): The first ONNX graph
            graph2 (OnnxGraph): The second ONNX graph
            io_map (list of pairs of string): The pairs of names [(out0/in0), (out1/in0), ...]
                                                representing outputs of the first graph and inputs of the second
                                                to be connected
            graph_name (str): name of the combined graph.
                              If not provided, the name is graph1.name and graph2.name connected by "_"

        Returns:
            graph (OnnxGraph): Combined graph

        """

        g_name = graph_name if graph_name else "_".join([graph1.name, graph2.name])
        graph = OnnxGraph(g_name)
        graph.nodes.extend(graph1.nodes)
        g2_node_begin = len(graph.nodes)
        graph.nodes.extend(graph2.nodes)
        g2_node_end = len(graph.nodes)

        io_map_g1_outs = {io[0] for io in io_map}
        io_map_g2_ins = {io[1] for io in io_map}
        reversed_io_map = {in_name: out_name for out_name, in_name in io_map}

        # connecting outputs of the first graph with the inputs of the second
        for node_idx in range(g2_node_begin, g2_node_end):
            node = graph.nodes[node_idx]
            for idx, name_ in enumerate(node.inputs):
                if name_ in reversed_io_map:
                    node.inputs[idx] = reversed_io_map.get(name_)

        # add inputs and outputs
        graph.inputs.extend(graph1.inputs)
        graph.inputs.extend([inp for inp in graph2.inputs if inp.name not in io_map_g2_ins])

        graph.outputs.extend([out for out in graph1.outputs if out.name not in io_map_g1_outs])
        graph.outputs.extend(graph2.outputs)

        # add initializers
        graph.initializers.extend(graph1.initializers)
        graph.initializers.extend([ini for ini in graph2.initializers if ini.name not in io_map_g2_ins])

        # add value_infos
        graph.value_infos.extend(graph1.value_infos)
        graph.value_infos.extend([value_info for value_info in graph2.value_infos if value_info not in io_map_g2_ins])

        # update g.node_map, g.prev_map and next_map
        graph.update_map()

        return graph

    @classmethod
    def parse(cls, path_or_bytes: Union[str, ModelProto, GraphProto], add_name_suffix: bool = False) -> 'OnnxGraph':
        if isinstance(path_or_bytes, str):
            check_input_file_path(path_or_bytes, file_max_size=ONNX_MODEL_MAX_SIZE)
            onnx_model = onnx.load(path_or_bytes)
        if isinstance(path_or_bytes, ModelProto):
            onnx_model = path_or_bytes
        if isinstance(path_or_bytes, GraphProto):
            onnx_graph = path_or_bytes
            meta = {}
        else:
            onnx_graph = onnx_model.graph
            meta = {
                'ir_version': onnx_model.ir_version,
                'domain': onnx_model.domain,
                'model_version': onnx_model.model_version,
                'doc_string': onnx_model.doc_string,
                'opset_imports': onnx_model.opset_import,
            }

        inputs = [OnnxPlaceHolder.parse(i) for i in onnx_graph.input]
        outputs = [OnnxPlaceHolder.parse(opt) for opt in onnx_graph.output]
        initializers = [OnnxInitializer.parse(i) for i in onnx_graph.initializer]

        nodes = []
        useless_value_infos = set()
        for node in onnx_graph.node:
            if node.op_type == 'Constant':
                initializers.append(OnnxInitializer.parse(node))
                useless_value_infos.add(node.output[0])
            else:
                nodes.append(OnnxNode.parse(node, add_name_suffix))

        value_infos = []
        for value_info in onnx_graph.value_info:
            if value_info.name not in useless_value_infos:
                value_infos.append(OnnxPlaceHolder.parse(value_info))

        graph = cls(onnx_graph.name, nodes, inputs, outputs, initializers, value_infos, **meta)
        return graph

    @classmethod
    def check_overlapping_names(
            cls, graph1: 'OnnxGraph', graph2: 'OnnxGraph', io_map: Optional[List[Tuple[str, str]]]
    ) -> List[Tuple[str, str]]:
        """Check whether there are name collisions between two graphs

        Arguments:
            graph1 (OnnxGraph): First graph
            graph2 (OnnxGraph): Second graph
            io_map (List): pairs of output/inputs to be connected.
                           If provided, overlapping present in the io_map argument will be ignored.

        Returns:
             a list of tuples where the first element represents the member containing overlapping names
            (One of: `node`, `edges`, `initializer`), and the second element contains a list of names
            that appear in both graphs on that category

        Optionally, it takes an io_map, representing the output/inputs to be connected.

        """

        if not isinstance(graph1, OnnxGraph):
            raise TypeError("g1 argument is not OnnxGraph")
        if not isinstance(graph2, OnnxGraph):
            raise TypeError("g2 argument is not OnnxGraph")

        def _overlapping(c1: List[str], c2: List[str]) -> List[str]:
            return list(set(c1) & set(c2))

        def _edge_names(graph: OnnxGraph, exclude: Optional[Set[str]] = None) -> List[str]:
            if not exclude:
                exclude = set()
            edges = []
            for node in graph.nodes:
                edges.extend(filter(lambda x: x and x not in exclude, node.inputs))
                edges.extend(filter(lambda x: x and x not in exclude, node.outputs))
            return edges

        result = []
        if not io_map:
            io_map = []
        io_map_inputs = {elem[1] for elem in io_map}

        # check name collisions for nodes, edges and initializers
        overlap = _overlapping([node.name for node in graph1.nodes], [node.name for node in graph2.nodes])
        if overlap:
            result.append(("nodes", overlap))

        overlap = _overlapping(_edge_names(graph1), _edge_names(graph2, exclude=io_map_inputs))
        if overlap:
            result.append(("edges", overlap))

        overlap = _overlapping([ini.name for ini in graph1.initializers], [ini.name for ini in graph2.initializers])
        if overlap:
            result.append(("initializer", overlap))

        return result

    @classmethod
    def add_prefix_graph(
            cls, graph: 'OnnxGraph', prefix: str, inplace: Optional[bool] = False,
            name_map: Optional[Dict[str, str]] = None
    ) -> 'OnnxGraph':
        """Adds a prefix to names of elements in a graph: nodes, edges, inputs, outputs,
        initializers and value infos.

        It can be used as a utility before merging graphs that have overlapping names.
        Empty names are not prefixed.

        Arguments:
            graph (OnnxGraph): Graph
            prefix (str): Prefix to be added to each name in the graph
            rename_nodes (bool): Whether to prefix node names
            ...
            inplace (bool): If true, modify the graph directly.
                            Otherwise, a copy will be created
            name_map (Dict): shared name_map in subgraph

        Returns:
            OnnxGraph

        """

        if not isinstance(graph, OnnxGraph):
            raise TypeError("graph argument is not OnnxGraph")

        g = copy.deepcopy(graph) if not inplace else graph

        def _prefix(prefix: str, name: str) -> str:
            return prefix + name if len(name) > 0 else name

        if not name_map:
            name_map = dict()

        # store prefixed names
        for node in graph.nodes:
            name_map.update({e: _prefix(prefix, e) for e in node.inputs})
            name_map.update({e: _prefix(prefix, e) for e in node.outputs})

        name_map.update({e.name: _prefix(prefix, e.name) for e in graph.inputs})
        name_map.update({e.name: _prefix(prefix, e.name) for e in graph.outputs})

        name_map.update({e.name: _prefix(prefix, e.name) for e in graph.nodes})
        name_map.update({e.name: _prefix(prefix, e.name) for e in graph.initializers})

        # add prefixes to all names
        for node in g.nodes:
            if node.name in name_map:
                node.name = name_map.get(node.name)
            for idx, inp in enumerate(node.inputs):
                if inp in name_map:
                    node.inputs[idx] = name_map.get(inp)
            for idx, out in enumerate(node.outputs):
                if out in name_map:
                    node.outputs[idx] = name_map.get(out)

        for elem in list(itertools.chain(g.inputs, g.outputs, g.initializers)):
            if elem.name in name_map:
                elem.name = name_map.get(elem.name)

        return g

    @classmethod
    def concat_graph(
            cls,
            graph1: 'OnnxGraph',
            graph2: 'OnnxGraph',
            io_map: List[Tuple[str, str]],
            prefix: str = "pre_",
            graph_name: Optional[str] = None,
    ) -> 'OnnxGraph':
        """Combine two ONNX graphs into a single one.

        The combined graph is defined by connecting the specified set of outputs/inputs. Those inputs/outputs
        not specified in the io_map argument will remain as inputs/outputs of the combined map.

        Arguments:
             graph1 (OnnxGraph): First graph
             graph2 (OnnxGraph): Second graph
             io_map (list of pairs of string): The pairs of names [(out0/in0), (out1/in0), ...]
                                                representing outputs of the first graph and inputs of the second
                                                to be connected
             prefix (string): Optional prefix to be added to all names in a graph.
                              By default, a string of `pre_` will be added to all names in g1 if necessary
             graph_name (string): Optional string for the combined graph
                                  By default, the name is g1.name and g2.name concatenated with an underscore delimiter

        Returns:
             OnnxGraph

        """

        if not isinstance(graph1, OnnxGraph):
            raise TypeError("g1 argument is not an ONNX graph")
        if not isinstance(graph2, OnnxGraph):
            raise TypeError("g2 argument is not an ONNX graph")

        # check for name collisions
        overlapping_names = cls.check_overlapping_names(graph1, graph2, io_map)
        if overlapping_names:
            category, names = overlapping_names[0]
            logger.warning(
                "Cant merge two graphs with overlapping names. " f"Found repeated {category} names：" + ",".join(names)
            )
            logger.info(f"A prefix `{prefix}` will be added to graph1")

            graph1 = cls.add_prefix_graph(graph1, prefix=prefix)

            io_map = [(prefix + io[0], io[1]) for io in io_map]

        g1_outs = {out.name for out in graph1.outputs}
        g2_ins = {inp.name for inp in graph2.inputs}

        # check input/output names specified in io_map argument are valid
        for g1_out_name, g2_in_name in io_map:
            if g1_out_name not in g1_outs:
                raise ValueError(f"Output {g1_out_name} is not present in g1")
            if g2_in_name not in g2_ins:
                raise ValueError(f"Input {g2_in_name} is not present in g2")

        # connecting two graphs
        graph = OnnxGraph.connect_graph(graph1, graph2, io_map, graph_name)

        return graph

    def add_input(self, name: str, dtype: str, shape: Sequence[Union[int, str]]) -> OnnxPlaceHolder:
        dtype = np.dtype(dtype)
        graph_input = OnnxPlaceHolder(name, dtype, shape)
        return self._add_input(graph_input)

    def add_output(self, name: str, dtype, shape) -> OnnxPlaceHolder:
        dtype = np.dtype(dtype)
        graph_output = OnnxPlaceHolder(name, dtype, shape)
        return self._add_output(graph_output)

    def add_initializer(self, name: str, value: np.ndarray) -> OnnxInitializer:
        initializer = OnnxInitializer(name, value)
        return self._add_initializer(initializer)

    def add_node(
            self,
            node_info: Dict[str, str],  # name, op_type合并后的参数
            inputs: Optional[List[str]] = None,
            outputs: Optional[List[str]] = None,
            attrs: Optional[Dict[str, object]] = None,
            domain: str = '',
    ) -> OnnxNode:
        name = node_info.get('name')
        op_type = node_info.get('op_type')

        node = OnnxNode(name, op_type, inputs, outputs, attrs=attrs, domain=domain)
        self.update_map()
        return self._add_node(node)

    def proto(self) -> GraphProto:
        self.toposort()
        return helper.make_graph(
            nodes=[node.proto() for node in self._nodes],
            name=self.name,
            inputs=[input.proto() for input in self._inputs],
            outputs=[output.proto() for output in self._outputs],
            initializer=[ini.proto() for ini in self._initializers],
            value_info=[val.proto() for val in self._value_infos],
        )

    def model(self) -> ModelProto:
        return helper.make_model(self.proto(), **self._meta)

    def save(self, path: str,
             save_as_external_data: bool = False,
             all_tensors_to_one_file: bool = True) -> None:

        # Threshold set to 1.9GB (instead of 2GB) due to calculation differences
        threshold = 1.9 * 1024 * 1024 * 1024

        try:
            serialized_model = self.model().SerializeToString()
            model_size = len(serialized_model)
            # Save as external data if model_size exceeds the threshold
            if model_size > threshold:
                save_as_external_data = True
        except ValueError:
            # Save as external data if model_size is too large and raises a ValueError
            save_as_external_data = True

            # Remove duplicate data file when saving as a single external data file
        base_name = os.path.basename(path) + '.data'
        file_name = os.path.join(os.path.dirname(path), base_name)
        if os.path.exists(file_name):
            os.remove(file_name)

        onnx.save(
            self.model(),
            path,
            save_as_external_data=save_as_external_data,
            all_tensors_to_one_file=all_tensors_to_one_file,
            location=os.path.basename(path) + '.data',
        )

    def infer_shape(self) -> None:
        # clear value_infos
        self._value_infos = []
        self._value_map = {}
        model = self.model()

        try:
            inferred_model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        except ValueError as e:
            with tempfile.TemporaryDirectory() as tmpdirname:
                onnx.save(model, os.path.join(tmpdirname, 'model.onnx'), save_as_external_data=True)
                onnx.shape_inference.infer_shapes_path(
                    os.path.join(tmpdirname, 'model.onnx'), os.path.join(tmpdirname, 'inferred_model.onnx')
                )
                infer_model_path = os.path.join(tmpdirname, 'inferred_model.onnx')
                if not Rule.input_file().max_size(ONNX_MODEL_MAX_SIZE).check(infer_model_path):
                    logger.error("Load inferred model failed")
                    raise OSError from e
                Rule.input_file().check(infer_model_path, will_raise=True)
                inferred_model = onnx.load(infer_model_path)

        # update value_infos
        graph = inferred_model.graph
        self._value_infos = [OnnxPlaceHolder.parse(v) for v in graph.value_info]
        self._value_map = {v.name: v for v in self._value_infos}

    def extract(
            self,
            new_model_save_path: str,
            input_name_list: List[str],
            output_name_list: List[str],
            enable_model_check: bool = True,
    ) -> 'OnnxGraph':

        def check_model(model):
            pass

        if not enable_model_check:
            onnx.checker.check_model = check_model

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.save(os.path.join(tmpdirname, 'model.onnx'))
            logger.info('Begin to extract the model.')
            try:
                onnx.utils.extract_model(
                    os.path.join(tmpdirname, 'model.onnx'), new_model_save_path, input_name_list, output_name_list
                )
            except ValueError as e:
                raise RuntimeError('Function extract() does not support a Large ONNX Model >2GB currently.') from e
            logger.info('Extract the model completed, model saved in {}.'.format(new_model_save_path))
        return OnnxGraph.parse(new_model_save_path)

    def extract_subgraph(
            self,
            start_node_names: List[str] = None,
            end_node_names: List[str] = None,
            subgraph_path: str = None,
            is_check_subgraph: bool = False,
            input_info: dict = None,  # input_shape, input_dtype合并后的参数
    ):

        if input_info is not None:
            input_shape = input_info.get('shape')
            input_dtype = input_info.get('dtype')
        else:
            input_shape = None
            input_dtype = None

        # do shape info by default
        try:
            self.infer_shape()
        except Exception as exp:
            logger.debug("Infer shape failed: %s", exp)

        # construct start nodes and/or end nodes set by default
        if not start_node_names:
            start_node_names = []
            for inp in self.inputs:
                start_node_names += [node.name for node in self.get_next_nodes(inp.name)]

        if not end_node_names:
            end_node_names = []
            for out in self.outputs:
                end_node_names.append(self.get_prev_node(out.name).name)

        # parse input info from input shape and input dtype
        input_shape_dict = self._parse_input_info(input_shape)
        input_dtype_dict = self._parse_input_info(input_dtype)

        all_node_names = {node.name for node in self.nodes}
        for start_node_name in start_node_names:
            if start_node_name not in all_node_names:
                raise ValueError(f'Start node {start_node_name} is not in this model')
        for end_node_name in end_node_names:
            if end_node_name not in all_node_names:
                raise ValueError(f'End node {end_node_name} is not in this model')

        input_name_list = []
        for start_node_name in start_node_names:
            start_node = self.get_node(start_node_name, node_type=Node)
            for inp in start_node.inputs:
                if len(inp) == 0:
                    continue
                if not self.get_node(inp, Initializer) and (inp not in input_name_list):
                    input_name_list.append(inp)

        output_name_list = []
        for end_node_name in end_node_names:
            end_node = self.get_node(end_node_name, node_type=Node)
            for oup in end_node.outputs:
                if oup not in output_name_list:
                    output_name_list.append(oup)

        start_nodes = [self.get_node(start_name, node_type=Node) for start_name in start_node_names]
        end_nodes = [self.get_node(end_name, node_type=Node) for end_name in end_node_names]

        top_down_visited = self._bfs_search_reachable_nodes(start_nodes)
        bottom_up_visited = self._bfs_search_reachable_nodes(end_nodes, top_down=False)
        reachable_nodes = top_down_visited & bottom_up_visited

        if not reachable_nodes:
            raise ValueError('There is no path from start nodes to end nodes')

        # collect reachable initializers and value_infos
        initializers = []
        value_infos = []
        for node in reachable_nodes:
            for inp in node.inputs:
                ini = self.get_node(inp, Initializer)
                if ini and ini not in initializers:
                    initializers.append(ini)
                elif self.get_prev_node(inp) not in reachable_nodes and inp not in input_name_list:
                    if len(inp) == 0:
                        continue
                    input_name_list.append(inp)
                elif self.get_node(inp, PlaceHolder) and inp not in input_name_list:
                    value_infos.append(self.get_node(inp, PlaceHolder))

        # remove isolated inputs
        valid_inputs = [
            inp
            for node in self.nodes
            for inp in node.inputs
        ]
        input_name_list = list(set(valid_inputs) & set(input_name_list))

        # check input shape and input dtype
        self._check_input_shape_and_dtype(input_name_list, input_shape_dict, input_dtype_dict)

        # add inputs and outputs for extracted graph
        inputs = self._add_new_io_placeholder(input_name_list, input_shape_dict, input_dtype_dict)
        outputs = self._add_new_io_placeholder(output_name_list)

        # save_model
        subgraph = OnnxGraph(
            'extracted graph', reachable_nodes, inputs, outputs, initializers, value_infos, **self._meta
        )
        subgraph.toposort()

        if subgraph_path and check_output_model_path(subgraph_path):
            subgraph.save(subgraph_path)
            logger.info('Extract the model completed, model saved in {}.'.format(subgraph_path))

        if is_check_subgraph:
            try:
                onnx.checker.check_model(subgraph.model())
            except Exception as exp:
                logger.info("Check subgraph failed, error is:", exp)

        return subgraph

    def simplify(self, **kwargs) -> 'OnnxGraph':
        try:
            from onnxsim import simplify
        except ImportError as err:
            raise RuntimeError("No module named 'onnxsim'") from err

        model = self.model()
        model_sim, check = simplify(model, **kwargs)
        if not check:
            raise RuntimeError("Simplified ONNX model could not be validated")

        return OnnxGraph.parse(model_sim)

    def _bfs_search_reachable_nodes(self, start_nodes, top_down=True):
        visited = set()
        queue = deque(start_nodes)
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if top_down:
                for output_name in node.outputs:
                    for next_node in self.get_next_nodes(output_name):
                        queue.append(next_node)
            else:
                for input_name in node.inputs:
                    prev_node = self.get_prev_node(input_name)
                    if prev_node:
                        queue.append(prev_node)
        return visited

    def _add_new_io_placeholder(self, name_list, input_shape_dict=None, input_dtype_dict=None):
        ph_list = []
        for name in name_list:
            value_info = self.get_node(name, PlaceHolder)
            ph_shape = None
            ph_dtype = np.dtype('float32')
            if value_info:
                ph_shape = value_info.shape
                ph_dtype = value_info.dtype
            if input_shape_dict and input_shape_dict.get(name):
                ph_shape = [int(i) for i in input_shape_dict[name]]
            if input_dtype_dict and input_dtype_dict.get(name):
                ph_dtype = np.dtype(input_dtype_dict[name])

            if ph_shape:
                onnx_placeholder = OnnxPlaceHolder(name, ph_dtype, ph_shape)
            else:
                onnx_placeholder = OnnxPlaceHolder(name, ph_dtype)
            ph_list.append(onnx_placeholder)
        return ph_list

    def _parse_input_info(self, input_info):
        input_info_dict = {}
        if not input_info:
            return input_info_dict

        input_segs = input_info.strip().split(";")
        for items in input_segs:
            input_field, input_value = items.strip().split(":")
            input_field = input_field.strip()
            input_value = [i.strip() for i in input_value.strip().split(",")]
            input_info_dict[input_field] = input_value

        return input_info_dict

    def _check_input_shape_and_dtype(self, input_name_list, input_shape_dict, input_dtype_dict):
        dtype_converter = {
            'bool': 'bool',
            'int': 'int32',
            'intc': 'int32',
            'intp': 'int32',
            'int8': 'int8',
            'int16': 'int16',
            'int32': 'int32',
            'int64': 'int64',
            'uint8': 'uint8',
            'uint16': 'uint16',
            'uint32': 'uint32',
            'uint64': 'uint64',
            'float': 'float64',
            'float16': 'float16',
            'float32': 'float32',
            'float64': 'float64',
            'complex': 'complex128',
            'complex64': 'complex64',
            'complex128': 'complex128',
            'fp16': 'float16',
            'fp32': 'float32',
            'fp64': 'float64',
        }

        for inp in input_shape_dict.keys():
            if inp not in input_name_list:
                logger.warning(
                    f'Input : {inp} is not in the inputs of the subgraph'
                    f'Please check it or the default shape will be applied.'
                )

        for inp, inp_dtype in input_dtype_dict.items():
            if inp not in input_name_list:
                logger.warning(
                    f'Input : {inp} is not in the inputs of the subgraph'
                    f'Please check it or the default dtype (float32) will be applied.'
                )
            if inp_dtype[0] not in dtype_converter:
                raise ValueError(f"The input type {inp_dtype} of {inp} is not valid. Please check it.")

            input_dtype_dict[inp] = dtype_converter[inp_dtype[0]]

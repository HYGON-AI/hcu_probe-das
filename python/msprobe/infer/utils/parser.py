# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from abc import abstractmethod
from msprobe.infer.utils.install import get_entry_points, INSTALL_INFO_MAP, build_extra, \
                                     install_tools, check_tools, download_comps
from msprobe.infer.utils.constants import MIND_STUDIO_LOGO

ALL_SUB_TOOLS = [pkg.get("arg-name") for pkg in INSTALL_INFO_MAP]
ALL_SUB_TOOLS_WITH_ALL = ["all"]
ALL_SUB_TOOLS_WITH_ALL.extend(ALL_SUB_TOOLS)


class AitCommand:
    def __init__(self, name, help_info="", alias_name="", group="Command") -> None:
        self.name = name
        self.help_info = help_info
        self.alias_name = alias_name
        self.group = group

    @abstractmethod
    def register_parser(self, parser):
        pass

    @abstractmethod
    def handle(self, args):
        pass


class BaseCommand(AitCommand):
    def __init__(self, name, help_info, children=None, alias_name="", group="Command") -> None:
        super().__init__(name, help_info, alias_name, group)

        self.parser = None
        self.children: list[BaseCommand] = []

        if isinstance(children, str):
            self.children = LazyEntryPointCommand.build_lazy_tasks(children)
        elif isinstance(children, list):
            for child in children:
                if isinstance(child, BaseCommand):
                    self.children.append(child)
                elif isinstance(child, str):
                    self.children.extend(LazyEntryPointCommand.build_lazy_tasks(child))
                else:
                    raise ValueError("unknow child")
        else:
            pass

    def register_parser(self, parser: argparse.ArgumentParser):
        self.parser = parser
        self.add_arguments(parser)
        parser.set_defaults(handle=self.handle)

        if not self.children:
            return

        subparsers = parser.add_subparsers(title="Command")
        # groups, now put it together. not real group
        groups = {"Command": []}
        for command in self.children:
            groups.setdefault(command.group, [])
            groups.get(command.group).append(command)
        for _, command_list in groups.items():
            for command in command_list:
                subparser = subparsers.add_parser(
                    command.name,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    help=command.help_info,
                    aliases=[command.alias_name] if command.alias_name else [],
                    description=command.help_info + " " + MIND_STUDIO_LOGO,
                )
                command.register_parser(subparser)

    def add_arguments(self, parser):
        pass

    def handle(self, _):
        if self.parser is not None:
            self.parser.print_help()


class LazyEntryPointCommand(AitCommand):
    def __init__(self, name, help_info, entry_point) -> None:
        super().__init__(name, help_info)
        self.entry_point = entry_point
        self.inner_task = None

    @staticmethod
    def build_lazy_tasks(entry_points_name):
        entry_points = get_entry_points(entry_points_name)
        tasks = []
        for entry_point in entry_points:
            entry_info = entry_point.name.split(":", 1)
            if len(entry_info) == 1:
                name, help_info = entry_info[0], ""
            else:
                name, help_info = entry_info

            tasks.append(LazyEntryPointCommand(name, help_info, entry_point))
        return tasks


    def register_parser(self, parser: argparse.ArgumentParser):
        self.register_parser_lazy(parser)

    def register_parser_lazy(self, parser: argparse.ArgumentParser):
        ori_parse_args = parser.parse_known_args

        def hook_parse_args(args=None, namespace=None):
            self.load_register_inner_task(parser)
            return ori_parse_args(args, namespace)

        parser.parse_known_args = hook_parse_args

    def load_register_inner_task(self, parser: argparse.ArgumentParser):
        self.inner_task: BaseCommand = self.entry_point.load()()
        if isinstance(self.inner_task, BaseCommand):
            self.inner_task.register_parser(parser)


class AitInstallCommand(BaseCommand):
    def __init__(self) -> None:
        super().__init__("install", "install msit tools", group="Install Command")

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "comp_names",
            default=None,
            nargs="+",
            choices=ALL_SUB_TOOLS_WITH_ALL,
            help="component's name",
        )

        parser.add_argument(
            "--find-links", "-f",
            default=None,
            type=str,
            help="the dir look for archives",
        )

        parser.add_argument(
            "--no-check",
            dest="no_check",
            action='store_true',
            help="whether check certificate while install benchmark",
        )

    def handle(self, args):
        if args.no_check:
            os.environ['NO_CHECK_CERTIFICATE'] = 'true'
        install_tools(args.comp_names, args.find_links)


class AitCheckCommand(BaseCommand):
    def __init__(self) -> None:
        super().__init__("check", "check msit tools status.", group="Install Command")

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "comp_names",
            default=None,
            nargs="+",
            choices=ALL_SUB_TOOLS_WITH_ALL,
            help="component's name",
        )

    def handle(self, args):
        check_tools(args.comp_names)


class AitBuildExtraCommand(BaseCommand):
    def __init__(self) -> None:
        super().__init__("build-extra", "build msit tools extra", group="Install Command")

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "comp_name",
            default=None,
            choices=ALL_SUB_TOOLS,
            help="component's name",
        )

        parser.add_argument(
            "--find-links", "-f",
            default=None,
            type=str,
            help="the dir look for archives",
        )

    def handle(self, args):
        build_extra(args.comp_name, args.find_links)


class DownloadCommand(BaseCommand):
    def __init__(self) -> None:
        super().__init__("download", "download packages", group="Install Command")

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "comp_names",
            default=None,
            choices=ALL_SUB_TOOLS_WITH_ALL,
            help="component's name",
        )

        parser.add_argument(
            "--dest", "-d",
            default=None,
            required=True,
            type=str,
            help=" Download packages into <dir>.",
        )

    def handle(self, args):
        download_comps(args.comp_names, args.dest)

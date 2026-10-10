# Third-Party Notices

This file records the source URL, pinned version/commit, copyright, license, local paths, and modification status of third-party code and documentation that is **contained in, derived from, or referenced at build time by** the hcu_probe-das (hcuprobe) repository.

## msProbe

This repository is an HCU-platform adaptation fork of Huawei msProbe; the vast majority of the code and documentation in this repository comes from upstream.

- **Source URL**: <https://gitcode.com/Ascend/msprobe> (branch `26.2.0`)
- **Pinned commit**: `76f701cd74b765b98e9652fa0730b377daa25559` (2026-09-23, fork baseline)
  - The corresponding baseline commit in this repository's git history is `87dec7eb1602b0b123c312eab0840b1d5ab1fd89` (hash differs due to history cleanup; the tree content is identical to the upstream baseline)
- **Copyright**: Copyright (c) Huawei Technologies Co., Ltd. (see individual file headers for the years)
- **License**: Code is under the [Mulan Permissive Software License, Version 2 (Mulan PSL v2)](LICENSE); a few files carry an Apache License 2.0 header (copyright likewise Huawei), in which case the file header prevails; upstream documentation is under [CC BY 4.0](docs/LICENSE)
- **Local paths (registered by scope)**:

  | Path in this repo | Upstream path | Notes |
  |---|---|---|
  | `python/hcuprobe/` | `python/msprobe/` | Main Python package, directory renamed as a whole |
  | `ccsrc/`, `cmake/`, `CMakeLists.txt`, `build.py`, `build.sh`, `hatch_build.py`, `MANIFEST.in` | Same paths | C++ extensions and build system |
  | `plugins/tb_graph_ascend/` | Same path | TensorBoard visualization plugins (hierarchy / monvis) |
  | `scripts/`, `examples/`, `pre-commit/`, `.devcontainer/`, `.vscode/`, and other project configuration | Same paths | Auxiliary scripts and project configuration |
  | `test/` | Same path | HCU adaptation on top of the upstream test cases |
  | `docs/zh/`, `docs/en/`, `docs/index.md`, `docs/conf.py`, `docs/_static/`, `docs/_templates/`, `docs/requirements.txt` | Same paths under `docs/` | Upstream documentation, CC BY 4.0; copies/adaptations must retain attribution to msProbe |
  | `README_ORIGIN.md`, `README_ORIGIN_EN.md` | `README.md`, `README_EN.md` | Upstream READMEs preserved as-is |

- **Modification status**: **Modified.** All commits above the baseline are HYGON-side modifications, mainly including:
  - Full rename of the package and CLI namespace from `msprobe` to `hcuprobe`;
  - HCU platform adaptation, including integration adaptations for PyTorch, Megatron, and verl on the HCU platform;
  - Unit test adaptation;
  - `pyproject.toml` build configuration changes;
  - HCU-side original additions (not from upstream): `docs/annotated/`, `docs/offline_analysis/`, `docs/infer/`, `docs/FAQ.md`, `docs/UPSTREAM_SYNC.md`, the `.claude/` usage guides, etc.

  Modifications and additions: Hygon Information Technology Co., Ltd., likewise released under Mulan PSL v2. For version alignment and the re-sync procedure, see [docs/UPSTREAM_SYNC.md](docs/UPSTREAM_SYNC.md).

> Maintenance note: when upgrading the upstream version, adding vendored third-party code, or changing build dependencies, please update this file together with [docs/UPSTREAM_SYNC.md](docs/UPSTREAM_SYNC.md).

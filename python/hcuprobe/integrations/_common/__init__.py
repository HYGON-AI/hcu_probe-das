"""Framework-agnostic common layer for hcu_probe integrations.

Modules under this package MUST NOT depend on any specific training framework
(megatron / trl / lightning / ...). Per-framework `integrations/<framework>/`
namespaces translate framework-specific argument shapes into the keyword-only
APIs exposed here.
"""

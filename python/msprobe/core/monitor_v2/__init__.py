"""
monitor_v2: next-generation monitoring core for unifying
PyTorch / MindSpore monitor logic behind a single, clean abstraction.

Goal:
  - keep production behavior stable while iterating;
  - migrate mature logic from existing monitors in small steps;
  - enforce clearer layering and performance constraints.

Current structure:
  base.py
    - BaseMonitorV2: minimal shared monitor scaffold.
  trainer.py
    - TrainerMonitorV2: orchestrator for step gating and writer integration.
  factory.py
    - MonitorFactory: registry of framework-specific monitor classes.
  writer.py
    - CSVWriterV2: CSV output for per-step monitor data.
"""
from .trainer import TrainerMonitorV2

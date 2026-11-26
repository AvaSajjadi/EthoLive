"""EthoLive: causal real-time behavior classification from animal pose.

Main modules: model (pose network wrapper and behavior network), scope_model (feature steps and
model files), calms21 and flyvsfly (data), features / quality / filters (fixed feature pipeline),
classify and evaluate (frame branch and metrics), live_scoring (streaming inference), alerts (bout rule).
"""

__version__ = "0.4.0"

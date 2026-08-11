"""Guided-diffusion machinery ported from MoMADiff (itself based on
https://github.com/openai/guided-diffusion) for the unified text-to-pose path.

Only the code paths used by the unified model are retained (cosine/linear
schedules, x0-prediction MSE training, ancestral and DDIM sampling); the
retained code is a line-faithful port.
"""

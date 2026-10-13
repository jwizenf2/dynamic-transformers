"""Tool-calling evaluation for per-request layer routing.

Plan: docs/research/plan_tool_calling_routing.md.

Pure-Python pieces (schemas, Qwen call parsing, BFCL-style AST checking, data loading) live here
so they can be unit tested without MLX, downloads or model weights.
"""

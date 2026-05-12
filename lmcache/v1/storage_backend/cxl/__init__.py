# SPDX-License-Identifier: Apache-2.0
"""CXL-backed shared-memory KV cache tier for LMCache.

The pool layout, header, slot structs, region bitmap/descriptors, and
`cudaHostRegister` bootstrap live in this package. Concurrency and the
higher-level backend glue are added in later slices.
"""

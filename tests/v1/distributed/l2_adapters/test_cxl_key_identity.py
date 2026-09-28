# SPDX-License-Identifier: Apache-2.0
"""Tenant-identity tests for the CXL L2 adapter's index key derivation.

The CXL index addresses slots by a single u64 derived from an
``ObjectKey``. That u64 is the *only* thing the index probe compares
(``index.py``: ``line0.chunk_hash != needle`` -> keep probing), so any
``ObjectKey`` field folded out of it aliases in the shared pool.

These tests pin the fields that must never alias:

- ``kv_rank`` -- under TP>1 the serving engine emits one ObjectKey per
  rank differing *only* in this field (the token hash carries no rank),
  so folding it out serves rank 1 the rank-0 shard.
- ``model_name`` -- distinct models must not share a slot.
- ``cache_salt`` -- per-user isolation.

See ``docs/design/v1/distributed/l2_adapters/cxl_multi_tenant.md`` §3.1.
"""

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.store import object_key_to_chunk_hash

CONTENT = b"\xde\xad\xbe\xef\x01\x02\x03\x04"


def _metadata(world_size: int = 1, worker_id: int = 0) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test-model",
        world_size=world_size,
        local_world_size=world_size,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 4, 64),
        chunk_size=16,
    )


def _obj_key(
    chunk_hash: bytes = CONTENT,
    model_name: str = "test-model",
    kv_rank: int = 0,
    cache_salt: str = "",
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name=model_name,
        kv_rank=kv_rank,
        cache_salt=cache_salt,
    )


# ---------- the fields that must not alias ----------


def test_kv_rank_changes_index_hash():
    """Two TP ranks of the same chunk must not collide.

    This is the regression test for the TP>1 aliasing bug: the two
    ObjectKeys differ *only* in kv_rank, exactly as
    ``ipc_key_to_object_keys`` emits them for a TP=2 lookup.
    """
    rank0 = object_key_to_chunk_hash(_obj_key(kv_rank=0))
    rank1 = object_key_to_chunk_hash(_obj_key(kv_rank=1))
    assert rank0 != rank1


def test_model_name_changes_index_hash():
    """Identical content under two models must not collide."""
    a = object_key_to_chunk_hash(_obj_key(model_name="model-a"))
    b = object_key_to_chunk_hash(_obj_key(model_name="model-b"))
    assert a != b


def test_cache_salt_changes_index_hash():
    """Identical content for two users must not collide."""
    u1 = object_key_to_chunk_hash(_obj_key(cache_salt="user-1"))
    u2 = object_key_to_chunk_hash(_obj_key(cache_salt="user-2"))
    assert u1 != u2


def test_content_hash_changes_index_hash():
    """The content hash still participates, of course."""
    a = object_key_to_chunk_hash(_obj_key(chunk_hash=b"\x01" * 8))
    b = object_key_to_chunk_hash(_obj_key(chunk_hash=b"\x02" * 8))
    assert a != b


def test_all_tp_ranks_distinct():
    """A realistic TP=8 fan-out yields 8 distinct index keys."""
    hashes = {object_key_to_chunk_hash(_obj_key(kv_rank=r)) for r in range(8)}
    assert len(hashes) == 8


# ---------- stability and range ----------


def test_identical_keys_hash_identically():
    """The derivation is deterministic: same identity -> same slot.

    Without this a store and its subsequent lookup would never match.
    """
    assert object_key_to_chunk_hash(_obj_key()) == object_key_to_chunk_hash(_obj_key())


def test_hash_fits_u64():
    """The index stores this in a c_uint64; it must fit unsigned."""
    for kv_rank in (0, 1, 2**31, 2**32 - 1):
        h = object_key_to_chunk_hash(_obj_key(kv_rank=kv_rank))
        assert 0 <= h < 2**64


def test_long_content_hash_not_truncated():
    """A 32-byte digest (blake3/sha256) participates in full.

    The previous derivation truncated to the first 8 bytes, so two
    sha256 digests sharing a prefix would alias.
    """
    shared_prefix = b"\xaa" * 8
    a = object_key_to_chunk_hash(_obj_key(chunk_hash=shared_prefix + b"\x01" * 24))
    b = object_key_to_chunk_hash(_obj_key(chunk_hash=shared_prefix + b"\x02" * 24))
    assert a != b


def test_short_content_hash_accepted():
    """Hashes shorter than 8 bytes are still usable (no padding logic)."""
    h = object_key_to_chunk_hash(_obj_key(chunk_hash=b"\x01\x02"))
    assert 0 <= h < 2**64


# ---------- field framing ----------


def test_field_boundaries_are_unambiguous():
    """Concatenation-style collisions must not occur.

    Without a delimiter, model_name="ab" + salt="c" and
    model_name="a" + salt="bc" would hash identically.
    """
    a = object_key_to_chunk_hash(_obj_key(model_name="ab", cache_salt="c"))
    b = object_key_to_chunk_hash(_obj_key(model_name="a", cache_salt="bc"))
    assert a != b


# ---------- ObjectKey identity ----------


def test_object_keys_differ_across_tp_ranks():
    """The keys themselves compare unequal across TP ranks.

    The store keys its slot cache on the derived u64, but callers hold
    ObjectKeys — both levels must separate ranks or the L1/L2 bookkeeping
    would alias even before the index sees a hash.
    """
    k0 = _obj_key(kv_rank=0)
    k1 = _obj_key(kv_rank=1)
    assert k0 != k1
    assert hash(k0) != hash(k1)
    assert object_key_to_chunk_hash(k0) != object_key_to_chunk_hash(k1)

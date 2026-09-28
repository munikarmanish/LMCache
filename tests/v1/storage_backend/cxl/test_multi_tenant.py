# SPDX-License-Identifier: Apache-2.0
"""One CXL pool serving many tenants concurrently.

A "tenant" is a distinct ``(model_name, kv_rank, cache_salt)`` — a model,
a TP shard of one, or a user. The pool carries no model identity: attach
checks only compatibility, and separation is per slot via the tenant
digest stamped in ``line0.geom_hash``.

These tests cover what §3.2 of
``docs/design/v1/distributed/l2_adapters/cxl_multi_tenant.md`` specifies:
two tenants sharing a pool never read each other's bytes, including when
their index hashes collide outright.

See also ``tests/v1/distributed/l2_adapters/test_cxl_key_identity.py``,
which pins the key-derivation itself.
"""

# Standard
import ctypes
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.cxl.layout import GEOM_HASH_SIZE
from lmcache.v1.storage_backend.cxl.store import (
    CXLStore,
    CXLStoreConfig,
    object_key_to_chunk_hash,
    object_key_to_tenant_digest,
)

POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)
CHUNK_SIZE = 64 * 1024
PAYLOAD = 512


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(prefix="cxl-mt-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    s = CXLStore(
        CXLStoreConfig(
            dev_path=path,
            node_id=0,
            max_chunk_size_bytes=CHUNK_SIZE,
            region_size=REGION_SIZE,
            initialize=True,
            run_lock_manager=True,
            use_process_lock_manager=False,
        )
    )
    try:
        yield s
    finally:
        s.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _key(
    content: int,
    *,
    model_name: str = "model-a",
    kv_rank: int = 0,
    cache_salt: str = "",
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=content.to_bytes(8, "little"),
        model_name=model_name,
        kv_rank=kv_rank,
        cache_salt=cache_salt,
    )


def _obj(fill: int) -> TensorMemoryObj:
    data = torch.full((PAYLOAD,), fill, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([PAYLOAD]),
        dtype=torch.uint8,
        address=data.data_ptr(),
        phy_size=PAYLOAD,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(raw_data=data, metadata=meta, parent_allocator=None)


def _read(store: CXLStore, key: ObjectKey):
    buf = torch.empty(PAYLOAD, dtype=torch.uint8)
    n = store.read_into(key, buf.data_ptr(), buf.numel())
    return None if n == 0 else int(buf[0])


# ---------- tenants coexist ----------


@pytest.mark.parametrize(
    "other",
    [
        {"model_name": "model-b"},
        {"kv_rank": 1},
        {"cache_salt": "user-2"},
    ],
    ids=["model", "kv_rank", "cache_salt"],
)
def test_two_tenants_same_content_do_not_alias(store, other):
    """Identical content under two tenants stays two distinct chunks."""
    a = _key(0xA1)
    b = _key(0xA1, **other)

    store.put_batch([a], [_obj(0x11)])
    store.put_batch([b], [_obj(0x22)])

    assert _read(store, a) == 0x11
    assert _read(store, b) == 0x22


def test_one_tenants_eviction_leaves_the_other(store):
    """Removing one tenant's chunk must not disturb the other's."""
    a = _key(0xB1)
    b = _key(0xB1, model_name="model-b")
    store.put_batch([a], [_obj(0x33)])
    store.put_batch([b], [_obj(0x44)])

    assert store.remove(a)
    assert _read(store, a) is None
    assert _read(store, b) == 0x44


def test_many_tenants_interleaved(store):
    """Several models and ranks over the same content hashes."""
    tenants = [
        {"model_name": "m1", "kv_rank": 0},
        {"model_name": "m1", "kv_rank": 1},
        {"model_name": "m2", "kv_rank": 0},
        {"model_name": "m2", "kv_rank": 1, "cache_salt": "u"},
    ]
    keys = []
    for i, t in enumerate(tenants):
        for content in range(4):
            k = _key(0xC000 + content, **t)
            store.put_batch([k], [_obj(i * 16 + content)])
            keys.append((k, i * 16 + content))

    for k, expected in keys:
        assert _read(store, k) == expected


# ---------- the collision path ----------


def test_colliding_index_hashes_resolve_to_own_slots(store):
    """Two tenants forced onto one u64 must not read each other's bytes.

    A real u64 collision cannot be found by search, so this fabricates
    one: tenant B's chunk is written into the index under tenant A's
    ``chunk_hash`` while keeping B's own tenant digest. The probe must
    reject it on the digest and keep walking, so A reads A's bytes and
    B reads nothing at that slot.
    """
    a = _key(0xD1)
    b = _key(0xD1, model_name="model-b")
    # Precondition: distinct tenants really do get distinct digests.
    assert object_key_to_tenant_digest(a) != object_key_to_tenant_digest(b)

    store.put_batch([a], [_obj(0x55)])
    slot_a = store.slot_index_of(a)
    assert slot_a is not None

    # Forge the collision: stamp B's digest onto a *copy* of A's slot in
    # the next probe position, so both live on one chain under one hash.
    hash_a = object_key_to_chunk_hash(a)
    victim = (slot_a + 1) % store.pool.layout.index_slot_count
    src = store.pool.slots()[slot_a].line0
    dst = store.pool.slots()[victim].line0
    dst.chunk_hash = hash_a & 0xFFFFFFFFFFFFFFFF
    dst.chunk_offset = src.chunk_offset
    dst.chunk_len = src.chunk_len
    dst.fmt = src.fmt
    dst.owner_node_id = src.owner_node_id
    dst.generation = src.generation
    ctypes.memmove(dst.geom_hash, object_key_to_tenant_digest(b), GEOM_HASH_SIZE)
    dst.state = src.state

    # A still resolves to its own slot, not the forged one.
    assert store.slot_index_of(a) == slot_a
    assert _read(store, a) == 0x55

    # B's key hashes elsewhere entirely, so the forged slot is not a hit
    # for it either — the digest match alone is not sufficient.
    assert store.slot_index_of(b) is None


def test_tenant_digest_is_stamped_on_the_slot(store):
    """A committed slot carries its own tenant's digest, not the header's."""
    k = _key(0xE1, model_name="stamped", kv_rank=3, cache_salt="s")
    store.put_batch([k], [_obj(0x66)])

    slot_idx = store.slot_index_of(k)
    assert slot_idx is not None
    stored = bytes(store.pool.slots()[slot_idx].line0.geom_hash)
    assert stored == object_key_to_tenant_digest(k)
    # The header's copy is unused as of LAYOUT_VERSION 2.
    assert bytes(store.pool.header.geom_hash) == b"\x00" * GEOM_HASH_SIZE


# ---------- mixed chunk sizes ----------


def _obj_sized(fill: int, size: int) -> TensorMemoryObj:
    data = torch.full((size,), fill, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.uint8,
        address=data.data_ptr(),
        phy_size=size,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(raw_data=data, metadata=meta, parent_allocator=None)


def _read_sized(store: CXLStore, key: ObjectKey, size: int):
    buf = torch.empty(size, dtype=torch.uint8)
    n = store.read_into(key, buf.data_ptr(), buf.numel())
    return None if n == 0 else (n, int(buf[0]))


def test_tenants_with_different_chunk_sizes_share_one_pool(store):
    """Models of different geometry coexist without a declared chunk size.

    This is the end state of the design: nothing in the config names a
    size, and each tenant's class is created from the exact bytes it
    stores.
    """
    small = _key(0xF1, model_name="small-model")
    large = _key(0xF1, model_name="large-model")

    store.put_batch([small], [_obj_sized(0x77, 1024)])
    store.put_batch([large], [_obj_sized(0x88, 16384)])

    assert _read_sized(store, small, 1024) == (1024, 0x77)
    assert _read_sized(store, large, 16384) == (16384, 0x88)

    # Two distinct heap classes, sized exactly to what was stored.
    assert sorted(store.heaps.classes()) == [1024, 16384]


def test_same_geometry_shares_a_class_across_models(store):
    """Two models with equal chunk bytes share one class, not one slot."""
    a = _key(0xF2, model_name="model-a")
    b = _key(0xF2, model_name="model-b")

    store.put_batch([a], [_obj_sized(0x99, 2048)])
    store.put_batch([b], [_obj_sized(0xAA, 2048)])

    assert store.heaps.classes() == [2048]
    assert _read_sized(store, a, 2048) == (2048, 0x99)
    assert _read_sized(store, b, 2048) == (2048, 0xAA)


def test_eviction_returns_a_chunk_to_its_own_class(store):
    """A freed chunk goes back to the class it came from."""
    small = _key(0xF3, model_name="small-model")
    large = _key(0xF3, model_name="large-model")
    store.put_batch([small], [_obj_sized(0xBB, 1024)])
    store.put_batch([large], [_obj_sized(0xCC, 16384)])

    before = store.heaps.stats()
    assert store.remove(small)
    after = store.heaps.stats()

    assert after[1024].free_slots == before[1024].free_slots + 1
    assert after[16384].free_slots == before[16384].free_slots


# ---------- geometry agreement ----------


def test_same_model_different_geometry_does_not_alias(store):
    """A geometry mismatch is a miss, never a misread.

    Two nodes running the same model under different KV geometry (a
    dtype or head-count disagreement) must not read each other's
    chunks: the bytes would be interpreted wrongly. Salting the tenant
    digest with the geometry makes them simply not match.
    """
    k = _key(0x101, model_name="shared-model")

    store.set_geometry("shared-model", b"geometry-A")
    store.put_batch([k], [_obj(0x11)])
    assert _read(store, k) == 0x11

    # A peer declaring different geometry derives a different digest...
    assert object_key_to_tenant_digest(k, b"geometry-A") != object_key_to_tenant_digest(
        k, b"geometry-B"
    )

    # ...so it cannot reach this chunk. Re-salting this store stands in
    # for that peer: the chunk it just wrote becomes unreachable, which
    # is exactly what the mismatched node would observe.
    store._geometry["shared-model"] = b"geometry-B"  # noqa: SLF001 - see above
    assert _read(store, k) is None


def test_geometry_salt_is_optional(store):
    """A pool whose writers declare no geometry behaves as before."""
    k = _key(0x102, model_name="undeclared")
    store.put_batch([k], [_obj(0x22)])
    assert _read(store, k) == 0x22
    assert object_key_to_tenant_digest(k) == object_key_to_tenant_digest(k, b"")


def test_redeclaring_a_different_geometry_is_refused(store):
    """Changing a model's geometry mid-process would orphan its chunks."""
    store.set_geometry("m", b"geometry-A")
    store.set_geometry("m", b"geometry-A")  # idempotent
    with pytest.raises(ValueError, match="already declared"):
        store.set_geometry("m", b"geometry-B")


def test_geometry_is_scoped_per_model(store):
    """One model's geometry does not affect another's digests."""
    a = _key(0x103, model_name="model-a")
    b = _key(0x103, model_name="model-b")
    store.set_geometry("model-a", b"geometry-A")

    store.put_batch([a], [_obj(0x33)])
    store.put_batch([b], [_obj(0x44)])

    assert _read(store, a) == 0x33
    assert _read(store, b) == 0x44

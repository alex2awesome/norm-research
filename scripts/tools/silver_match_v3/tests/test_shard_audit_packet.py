from scripts.tools.silver_match_v3.shard_audit_packet import shard_id


def test_audit_shard_is_stable_and_in_range():
    assert shard_id("uid", 3) == shard_id("uid", 3)
    assert 0 <= shard_id("uid", 3) < 3

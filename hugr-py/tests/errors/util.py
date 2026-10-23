from syrupy.assertion import SnapshotAssertion


def error_snap(err: str, snap: SnapshotAssertion | None = None):
    if snap is not None:
        assert err == snap

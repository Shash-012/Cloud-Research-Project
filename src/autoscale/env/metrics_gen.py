"""Synthetic collectl-like metrics with explicit planted truth."""

METRIC_GROUP_SIZES = {
    "cpu": 21,
    "memory": 30,
    "buddy": 14,
    "disk": 9,
    "inode": 4,
    "network": 18,
    "tcp": 32,
}


def metric_names() -> tuple[str, ...]:
    return tuple(
        f"{group}_{index:02d}"
        for group, size in METRIC_GROUP_SIZES.items()
        for index in range(size)
    )

"""Warehouse Colour-Sort hackathon starter package.

Importing this package registers the WarehouseSort-v1 ManiSkill environment.
"""

from warehouse_sort import windows_cuda  # noqa: F401  (Windows GPU sim DLL shim)
from warehouse_sort.env import WarehouseSortEnv  # noqa: F401  (registers WarehouseSort-v1)

__all__ = ["WarehouseSortEnv"]

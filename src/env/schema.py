"""Data warehouse table & column schema catalog."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnMeta:
    """Metadata for a single column in a warehouse table."""

    name: str
    data_type: str
    comment: str = ""
    is_partition: bool = False


@dataclass(frozen=True)
class TableMeta:
    """Metadata for a warehouse table including layering and partition info."""

    name: str
    layer: str  # ods | dwd | dws | dim
    comment: str
    columns: tuple[ColumnMeta, ...]
    partition_col: str | None = None

    @property
    def is_partitioned(self) -> bool:
        return self.partition_col is not None

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


PARTITIONED_LAYER_PREFIXES = ("ods_", "dwd_", "dws_")
DEFAULT_PARTITION_COLS = ("ds", "dt")


@dataclass
class WarehouseCatalog:
    """Registry of all warehouse tables and partition definitions."""

    tables: dict[str, TableMeta] = field(default_factory=dict)

    def register(self, table: TableMeta) -> None:
        self.tables[table.name.lower()] = table

    def get(self, table_name: str) -> TableMeta | None:
        key = self._normalize_table_name(table_name)
        return self.tables.get(key)

    @staticmethod
    def _normalize_table_name(table_name: str) -> str:
        name = table_name.strip().strip("`").strip('"').lower()
        if "." in name:
            name = name.split(".")[-1]
        return name

    def is_partitioned(self, table_name: str) -> bool:
        key = self._normalize_table_name(table_name)
        meta = self.tables.get(key)
        if meta is not None:
            return meta.is_partitioned
        return key.startswith(PARTITIONED_LAYER_PREFIXES)

    def get_partition_col(self, table_name: str) -> str | None:
        key = self._normalize_table_name(table_name)
        meta = self.tables.get(key)
        if meta is not None:
            return meta.partition_col
        if key.startswith(PARTITIONED_LAYER_PREFIXES):
            return "ds"
        return None

    def list_tables(self) -> list[TableMeta]:
        return list(self.tables.values())

    def format_catalog_summary(self) -> str:
        """Compact summary of all tables, layers, partition columns, and column names."""
        lines: list[str] = []
        layer_order = {"ods": 0, "dwd": 1, "dws": 2, "dim": 3}
        ordered = sorted(
            self.tables.values(),
            key=lambda t: (layer_order.get(t.layer.lower(), 9), t.name),
        )
        for tbl in ordered:
            part_info = f"PARTITIONED BY ({tbl.partition_col})" if tbl.partition_col else "UNPARTITIONED"
            cols = ", ".join(f"{c.name}:{c.data_type}" for c in tbl.columns)
            lines.append(
                f"- [{tbl.layer.upper()}] {tbl.name} ({part_info}) -- {tbl.comment}\n"
                f"  columns: {cols}"
            )
        return "\n".join(lines)

    def format_table_ddl(self, table_names: list[str]) -> str:
        """Detailed schema & column comments for selected tables."""
        blocks: list[str] = []
        seen: set[str] = set()
        for raw in table_names:
            key = self._normalize_table_name(raw)
            if key in seen:
                continue
            seen.add(key)
            tbl = self.tables.get(key)
            if tbl is None:
                continue
            col_lines = []
            for c in tbl.columns:
                part_tag = " [PARTITION KEY]" if c.is_partition else ""
                col_lines.append(f"  {c.name} {c.data_type}{part_tag}  -- {c.comment}")
            part_clause = (
                f"PARTITIONED BY ({tbl.partition_col} STRING)"
                if tbl.partition_col
                else "-- Non-partitioned dimension table"
            )
            blocks.append(
                f"TABLE {tbl.name} ({tbl.layer.upper()}): {tbl.comment}\n"
                f"{part_clause}\n"
                + "\n".join(col_lines)
            )
        return "\n\n".join(blocks)

from .client import BolognaOpenData
from typing import Any, Dict, List, Optional
import re, html

# --------------
# list datasets
# --------------
async def list_catalog(
    client: BolognaOpenData,
    q: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, str]]:
    """
    List datasets

    Returns a list of dicts: {'dataset_id': str, 'title': str}

    Args:
        q: optional free-text search.
        limit: number of datasets to return (API max ~100).
    """

    catalog = await client.list_datasets(q=q, limit=limit)

    out: List[Dict[str, str]] = []
    # taking out title because model gets confused between title
    for item in catalog.get("results", []):
        dsid = item.get("dataset_id", "")
        out.append({"dataset_id": dsid})
    return out


# --------------
# preview dataset
# --------------
def _size_bytes(obj: Any) -> int:
    # robust: no JSON, just repr → bytes
    return len(repr(obj).encode("utf-8", errors="replace"))

def _shallow_truncate(v: Any, max_chars: int = 200, max_list_items: int = 5, max_dict_items: int = 10) -> Any:
    # primitives
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, (bytes, bytearray)):
        return (v[:max_chars] + b"...") if len(v) > max_chars else v
    if isinstance(v, str):
        return v if len(v) <= max_chars else v[:max_chars] + "…"

    # lists/tuples: keep only a few items, shallowly truncated
    if isinstance(v, (list, tuple)):
        n = len(v)
        keep = min(n, max_list_items)
        head = [_shallow_truncate(x, max_chars, max_list_items, max_dict_items) for x in v[:keep]]
        if n > keep:
            head.append(f"… (+{n-keep} more)")
        return head if isinstance(v, list) else tuple(head)

    # dicts: keep a few key/value pairs, shallowly truncated
    if isinstance(v, dict):
        items = list(v.items())
        keep = min(len(items), max_dict_items)
        kept = {k: _shallow_truncate(val, max_chars, max_list_items, max_dict_items) for k, val in items[:keep]}
        if len(items) > keep:
            kept["…"] = f"[+{len(items)-keep} more keys]"
        return kept

    # fallback: string repr capped
    s = repr(v)
    return s if len(s) <= max_chars else s[:max_chars] + "…"

async def preview_dataset(
    client,
    dataset_id: str,
    limit: int = 5,
    max_bytes: int = 20_000,      # payload budget
    max_chars: int = 200,         # per-cell cap
    max_list_items: int = 5,
    max_dict_items: int = 10,
) -> Dict[str, Any]:
    """
    Size-safe preview that stops when `max_bytes` would be exceeded.
    No JSON serialization is used; sizing is based on repr(...).
    """
    page = await client.query_records(dataset_id, limit=limit)
    rows = page.get("results", []) or []

    out: List[Dict[str, Any]] = []
    used = 0
    truncated_rows = False
    truncated_cells = False

    for r in rows:
        # shallow truncation for every cell
        safe_row = {k: _shallow_truncate(v, max_chars, max_list_items, max_dict_items) for k, v in r.items()}
        if safe_row != r:
            truncated_cells = True

        row_size = _size_bytes(safe_row)
        if used + row_size > max_bytes:
            truncated_rows = True
            break

        out.append(safe_row)
        used += row_size

    return {
        "rows": out,
        "meta": {
            "requested_limit": limit,
            "byte_budget": max_bytes,
            "bytes_used": used,
            "truncated_rows": truncated_rows,
            "truncated_cells": truncated_cells,
        },
    }

# ----------------
# get dataset description
# ----------------
def _html_to_text(s: str) -> str:
    if not s:
        return ""
    # normalize line breaks for common tags
    s = re.sub(r"<\s*(br|/p)\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<\s*p\s*>", "", s, flags=re.I)
    # drop all remaining tags
    s = re.sub(r"<[^>]+>", "", s)
    # unescape entities (&amp;, &egrave;, etc.)
    return html.unescape(s).strip()

async def get_dataset_description(
    client: BolognaOpenData,
    dataset_id: str
) -> str:
    """
    Return plain-text description from metas.default.description (HTML stripped).
    """
    meta = await client.get_dataset(dataset_id)
    raw = (
        meta.get("metas", {})
            .get("default", {})
            .get("description", "")
            or ""
    )
    return _html_to_text(raw)

# ----------------
# get dataset fields
# ----------------
async def get_dataset_fields(
    client: BolognaOpenData,
    dataset_id: str
) -> List[Dict[str, str]]:
    """
    Return the dataset fields (columns) as a minimal schema:
    [{'name': str, 'type': str, 'label': str}]
    """
    meta = await client.get_dataset(dataset_id)

    # Handle both shapes just in case:
    fields = meta.get("fields")
    if fields is None and isinstance(meta.get("dataset", {}), dict):
        fields = meta["dataset"].get("fields")

    out: List[Dict[str, str]] = []
    for f in fields or []:
        if not isinstance(f, dict):
            continue
        name = f.get("name", "")
        desc = f.get("description", "")
        ftype = (f.get("type") or "").lower()
        label = f.get("label") or name
        if name:
            out.append({"name": name, "description": desc, "type": ftype, "label": label})
    return out

# ----------------
# check if dataset is geo
# ----------------
GEO_TYPES = {"geo_point_2d", "geo_shape"}

async def is_geo_dataset(
    client: BolognaOpenData,
    dataset_id: str
) -> Dict[str, Any]:
    """
    Inspect dataset metadata and report geo capabilities.

    Returns:
      {
        "is_geo": bool,
        "geom_types": List[str],       # e.g., ["Point"] (from metas.default.geometry_types)
        "geom_fields": List[str],      # field names with type geo_point_2d / geo_shape
        "bbox": Optional[List[float]], # [west, south, east, north] if available
        "coordinate_order": "lon,lat", # Opendatasoft uses WGS84 lon/lat
      }
    """
    meta = await client.get_dataset(dataset_id)

    # 1) feature flag
    features = set((meta.get("features") or []))

    # 2) geo fields
    fields = meta.get("fields") or []
    geom_fields = [
        f.get("name")
        for f in fields
        if isinstance(f, dict) and (f.get("type") in GEO_TYPES)
    ]

    # 3) geometry types + bbox from metas.default
    default_meta = (meta.get("metas") or {}).get("default") or {}
    geom_types = list(default_meta.get("geometry_types") or [])

    bbox_list: Optional[List[float]] = None
    bbox_obj = default_meta.get("bbox")
    # bbox is a GeoJSON Feature with a Polygon; extract [minLon, minLat, maxLon, maxLat]
    try:
        if bbox_obj and bbox_obj.get("geometry") and bbox_obj["geometry"].get("coordinates"):
            coords = bbox_obj["geometry"]["coordinates"][0]  # outer ring
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            west, east = min(lons), max(lons)
            south, north = min(lats), max(lats)
            bbox_list = [west, south, east, north]
    except Exception:
        bbox_list = None  # not present or not parseable

    is_geo = ("geo" in features) or bool(geom_fields) or bool(geom_types)

    return {
        "is_geo": is_geo,
        "geom_types": geom_types,
        "geom_fields": geom_fields,
        "bbox": bbox_list,
        "coordinate_order": "lon,lat" if is_geo else None,   # Opendatasoft convention
    }

# ----------------
# get dataset time info
# ----------------
_TIME_TYPE_PREFERRED = {"date", "datetime"}
_TIME_NAME_CANDIDATES = ["anno", "year", "anno_rif", "data", "date", "timestamp"]

def _pick_time_column(fields: List[Dict[str, Any]]) -> Optional[str]:
    """
    Prefer fields with type date/datetime, else fall back to common names.
    """
    # 1) prefer by type
    for f in fields:
        if not isinstance(f, dict):
            continue
        ftype = (f.get("type") or "").lower()
        if ftype in _TIME_TYPE_PREFERRED:
            return f.get("name")
    # 2) fallback by common names (case-insensitive)
    lower_map = {(f.get("name") or "").lower(): f.get("name")
                 for f in fields if isinstance(f, dict)}
    for cand in _TIME_NAME_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand]
    return None

def _freq_label(url_or_label: Optional[str]) -> Optional[str]:
    """Compact label from DCAT frequency URIs (e.g., .../IRREG) or passthrough."""
    if not url_or_label:
        return None
    s = str(url_or_label)
    if "/" in s:
        return s.rstrip("/").split("/")[-1] or None
    return s

async def get_dataset_time_info(
    client: BolognaOpenData,
    dataset_id: str,
    time_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Combined temporal profile:
    - Derived data coverage via MIN/MAX on a detected time-like column (raw API strings)
    - Metadata timestamps & frequency

    Returns:
      {
        "time_col": str|None,
        "start": str|None,   # raw API value (e.g., '1986-01-01T00:00:00+00:00')
        "end": str|None,     # raw API value
        "fallback": bool,    # True if MIN/MAX failed
        "last_modified": str|None,
        "data_processed": str|None,
        "metadata_processed": str|None,
        "update_frequency": str|None,  # e.g., 'ANNUAL', 'IRREG'
        "records_count": int|None
      }
    """
    meta = await client.get_dataset(dataset_id)

    fields = meta.get("fields") or []
    metas  = meta.get("metas", {}) or {}
    dflt   = metas.get("default", {}) or {}
    dcat   = metas.get("dcat", {}) or {}

    # pick time column if not provided
    col = time_col or _pick_time_column(fields)

    start_val: Optional[str] = None
    end_val: Optional[str] = None
    fallback = True

    if col:
        # ODSQL supports MIN/MAX
        sel = f"MIN({col}) AS t_min, MAX({col}) AS t_max"
        try:
            res = await client.query_records(dataset_id, select=sel, limit=1)
            row = (res.get("results") or [{}])[0]
            t_min = row.get("t_min")
            t_max = row.get("t_max")
            if t_min is not None and t_max is not None:
                # keep raw strings/numbers as provided by API
                start_val = str(t_min)
                end_val   = str(t_max)
                fallback  = False
        except Exception:
            pass  # keep fallback=True

    freq = _freq_label(dcat.get("accrualperiodicity") or dflt.get("update_frequency"))

    return {
        "time_col": col,
        "start": start_val,
        "end": end_val,
        "fallback": fallback,
        "last_modified": dflt.get("modified"),
        "data_processed": dflt.get("data_processed"),
        "metadata_processed": dflt.get("metadata_processed"),
        "update_frequency": freq,
        "records_count": dflt.get("records_count"),
    }

# ----------------
# estimate dataset size
# ----------------
async def is_dataset_too_heavy(
    client: BolognaOpenData, 
    dataset_id: str, 
    threshold: int = 5_000_000  # 5MB
) -> bool:
    """
    Estimate dataset size based on record count and field count.
    
    Args:
        client: Bologna OpenData client
        dataset_id: Dataset identifier
        threshold: Size threshold in bytes (default: 5MB)
        
    Returns:
        True if dataset is too heavy, False otherwise
    """
    try:
        # Get dataset metadata
        dataset_info = await client.get_dataset(dataset_id)
        total_records = dataset_info.get("metas", {}).get("default", {}).get("records_count", 0)
        fields = dataset_info.get("fields", [])
        
        if total_records == 0:
            return False  # Empty dataset, not heavy
            
        # Estimate based on record count and field count
        # Conservative estimate: 2 bytes per field per record (for parquet compression)
        estimated_size = total_records * len(fields) * 2
        
        print(f"Dataset {dataset_id}: {total_records} records, {len(fields)} fields, estimated {estimated_size/ 1024 / 1024:.2f} MegaBytes")
        
        return estimated_size > threshold
        
    except Exception as e:
        # If we can't estimate, assume it's not too heavy and let it load
        print(f"Warning: Could not estimate size for {dataset_id}: {e}")
        return False

# ----------------
# export dataset as parquet
# ----------------
async def get_dataset_bytes(client: BolognaOpenData, dataset_id: str) -> bytes:
    """
    Download dataset as parquet bytes using the provided client (with HTTP/2 enabled).
    
    Args:
        client: BolognaOpenData client instance (with HTTP/2 enabled)
        dataset_id: Dataset identifier
        
    Returns:
        Raw parquet bytes
    """
    try:
        print(f"Starting download of dataset: {dataset_id}")
        # Use the provided client (which now has HTTP/2 enabled)
        parquet_bytes = await client.export(dataset_id, "parquet")
        print(f"Download completed. Size: {len(parquet_bytes)} bytes")
        return parquet_bytes
        
    except Exception as e:
        print(f"Error exporting dataset: {e}")
        raise
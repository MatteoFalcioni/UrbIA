# bologna_opendata.py
import httpx
from typing import Optional, Dict, Any

BASE_URL = "https://opendata.comune.bologna.it/api/explore/v2.1"


class BolognaOpenData:
    def __init__(self, timeout: float = 30.0):
        """
        Initialize the async HTTP client with connection pooling.

        Args:
            timeout: request timeout in seconds (default 30.0).
        """
        # Configure connection pooling for efficiency
        self._limits = httpx.Limits(
            max_keepalive_connections=10, max_connections=20, keepalive_expiry=60.0
        )

        # Create timeout config with longer read timeout for large downloads
        self._timeout_config = httpx.Timeout(
            connect=10.0,  # Connection timeout
            read=timeout,  # Read timeout (for large downloads)
            write=10.0,  # Write timeout
            pool=5.0,  # Pool timeout
        )

        # Don't create client at init - create on first use (lazy initialization)
        self._client: Optional[httpx.AsyncClient] = None
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """Check if the client is closed."""
        return self._closed or (
            self._client is not None
            and hasattr(self._client, "is_closed")
            and self._client.is_closed
        )

    async def close(self):
        """
        Close the underlying HTTP client.
        Must be called at the end of usage to free sockets.
        """
        if not self.is_closed and self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                # Ignore cleanup errors - connection might already be closed
                # This is expected behavior when the OS forcibly closes connections
                pass
            finally:
                self._closed = True

    async def _ensure_client_ready(self):
        """
        Ensure the client is ready for use, recreating if necessary.
        This handles cases where the client might be in an invalid state.
        """
        if self._client is None or self.is_closed:
            # Create or recreate client
            if self._client is not None:
                await self.close()  # Clean up old client

            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=self._timeout_config,
                limits=self._limits,
                http2=True,  # Enable HTTP/2 for better performance
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; LGUrbanBot/1.0; +https://github.com/matteofalcioni/LG-Urban)",
                    "Accept": "application/json, application/parquet, */*",
                },
            )
            self._closed = False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        if not self._closed:
            # This is a fallback - proper cleanup should use close() or context manager
            try:
                import asyncio

                if asyncio.get_event_loop().is_running():
                    # Can't run async code in destructor if loop is running
                    pass
                else:
                    asyncio.run(self._client.aclose())
            except Exception:
                pass

    async def list_datasets(
        self,
        q: Optional[str] = None,
        where: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Search the catalog (list datasets).

        This works at the *dataset level* (not rows inside a dataset).

        Args:
            q: (optional) search string. If provided, the client builds a
               safe ODSQL query using search('q').
               Example: q="residenti"
            where: (optional) raw ODSQL filter. Only needed if you want
                   precise catalog filters (e.g., theme='Ambiente').
                   In most cases you don't need this.
            limit: number of datasets to return (default 20).
            offset: pagination offset (default 0).

        Returns:
            A JSON dict with a "results" list. Each item includes
            dataset_id, metas.default.title, description, etc.

        Notes:
            - In Bologna's portal, q is internally translated to
              where=search('term').
            - The 'where' param here is rarely needed at catalog level,
              except for advanced filtering by theme/keyword.
        """
        params = {"limit": limit, "offset": offset}
        if where:
            params["where"] = where
        elif q:
            esc = q.replace("'", "''")
            params["where"] = f"search('{esc}')"

        await self._ensure_client_ready()
        try:
            r = await self._client.get("/catalog/datasets", params=params)
            r.raise_for_status()
            return r.json()
        except (RuntimeError, ConnectionError) as e:
            if "closed" in str(e).lower():
                # Client was closed, recreate and retry once
                await self._ensure_client_ready()
                r = await self._client.get("/catalog/datasets", params=params)
                r.raise_for_status()
                return r.json()
            raise

    async def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific dataset.

        Args:
            dataset_id: the id string from the catalog (e.g.,
                        "numero-di-residenti-per-quartiere").

        Returns:
            JSON dict containing dataset info, including schema/fields.
        """
        await self._ensure_client_ready()
        try:
            r = await self._client.get(f"/catalog/datasets/{dataset_id}")
            r.raise_for_status()
            return r.json()
        except (RuntimeError, ConnectionError) as e:
            if "closed" in str(e).lower():
                await self._ensure_client_ready()
                r = await self._client.get(f"/catalog/datasets/{dataset_id}")
                r.raise_for_status()
                return r.json()
            raise

    async def query_records(
        self,
        dataset_id: str,
        select: str = "*",
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Query rows from a dataset using ODSQL.

        Args:
            dataset_id: dataset id string.
            select: comma-separated list of columns (default "*").
            where: ODSQL filter, e.g. "quartiere='Navile' AND anno=2023".
            order_by: ODSQL order clause, e.g. "anno DESC".
            limit: number of rows to fetch (default 100).
            offset: pagination offset (default 0).

        Returns:
            JSON dict with a "results" list, each element a row (dict).

        Notes:
            This is the main method for slicing data by condition.
            Without a where/limit, you can easily request thousands
            of rows, so always filter.
        """
        params = {"select": select, "limit": limit, "offset": offset}
        if (
            where
        ):  # we won't use where in agentic workflows, we'll take it out when using this as a @tool
            params["where"] = where
        if order_by:
            params["order_by"] = order_by

        await self._ensure_client_ready()
        try:
            r = await self._client.get(
                f"/catalog/datasets/{dataset_id}/records", params=params
            )
            r.raise_for_status()
            return r.json()
        except (RuntimeError, ConnectionError) as e:
            if "closed" in str(e).lower():
                await self._ensure_client_ready()
                r = await self._client.get(
                    f"/catalog/datasets/{dataset_id}/records", params=params
                )
                r.raise_for_status()
                return r.json()
            raise

    async def export_to_file(self, dataset_id: str, path: str, fmt: str = "parquet") -> None:
        """
        Download the full dataset directly to a file using streaming.
        This avoids loading the entire dataset into memory.

        Args:
            dataset_id: dataset id string.
            path: file path to save to.
            fmt: format string (default "parquet").
        """
        await self._ensure_client_ready()
        
        async def _stream_download():
            async with self._client.stream(
                "GET", f"/catalog/datasets/{dataset_id}/exports/{fmt}"
            ) as response:
                response.raise_for_status()
                with open(path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)

        try:
            await _stream_download()
        except (RuntimeError, ConnectionError) as e:
            if "closed" in str(e).lower():
                await self._ensure_client_ready()
                await _stream_download()
            else:
                raise

    async def export(self, dataset_id: str, fmt: str = "parquet") -> bytes:
        """
        Download the full dataset in one file (no row limit).
        kept for backward compatibility.
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            await self.export_to_file(dataset_id, tmp_path, fmt)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

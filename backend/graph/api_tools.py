import json
from typing_extensions import Annotated
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from backend.opendata_api.helpers import (
    list_catalog,
    preview_dataset,
    get_dataset_description,
    get_dataset_fields,
    is_geo_dataset,
    get_dataset_time_info,
)
from backend.opendata_api.init_client import client



# --- TOOLS ---
@tool(
    name_or_callable="list_catalog",
    description="Search the dataset catalog with a keyword."
)
async def list_catalog_tool(
    q: Annotated[str, "The dataset search keyword"],
    runtime: ToolRuntime,
) -> Command:
    res = await list_catalog(client=client, q=q, limit=15)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(res, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool(
    name_or_callable="preview_dataset",
    description="Preview the first few rows of a dataset."
)
async def preview_dataset_tool(
    dataset_id: Annotated[str, "The dataset ID"],
    runtime: ToolRuntime,
) -> Command:
    res = await preview_dataset(client=client, dataset_id=dataset_id, limit=5)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(res, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool(
    name_or_callable="get_dataset_description",
    description="Get the human-written description of a dataset."
)
async def get_dataset_description_tool(
    dataset_id: Annotated[str, "The dataset ID"],
    runtime: ToolRuntime,
) -> Command:
    desc = await get_dataset_description(client=client, dataset_id=dataset_id)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=desc,
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool(
    name_or_callable="get_dataset_fields",
    description="Get the list of fields/columns in a dataset."
)
async def get_dataset_fields_tool(
    dataset_id: Annotated[str, "The dataset ID"],
    runtime: ToolRuntime,
) -> Command:
    fields = await get_dataset_fields(client=client, dataset_id=dataset_id)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(fields, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool(
    name_or_callable="is_geo_dataset",
    description="Check if a dataset has a `geo_point_2d` column."
)
async def is_geo_dataset_tool(
    dataset_id: Annotated[str, "The dataset ID"],
    runtime: ToolRuntime,
) -> Command:
    res = await is_geo_dataset(client=client, dataset_id=dataset_id)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(res, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool(
    name_or_callable="get_dataset_time_info",
    description="Get the temporal coverage of a dataset."
)
async def get_dataset_time_info_tool(
    dataset_id: Annotated[str, "The dataset ID"],
    runtime: ToolRuntime,
) -> Command:
    res = await get_dataset_time_info(client=client, dataset_id=dataset_id)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(res, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )

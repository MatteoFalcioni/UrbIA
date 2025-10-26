import folium
import tempfile
from typing import Optional, List
from typing_extensions import Annotated
from langgraph.types import Command
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from pathlib import Path
from folium.plugins import DualMap
from folium import Element

# Import the session manager
from backend.graph.dataset_tools import session_manager, get_session_key


# --------------------------
# Google Geocoding Tool (lazy loaded)
# --------------------------
_google_geocoding_tool = None

def get_google_geocoding_tool():
    """Lazy load Google Geocoding tool only when needed."""
    global _google_geocoding_tool
    if _google_geocoding_tool is None:
        from langchain_google_community.geocoding import GoogleGeocodingTool
        _google_geocoding_tool = GoogleGeocodingTool()
    return _google_geocoding_tool

#-------------------------------
# single ortofoto tool
#-------------------------------

def folium_ortho_fn(year, bbox=None, zoom_start=16, save_path: Optional[Path]=None):
    # Center on bbox or Piazza Maggiore-ish fallback
    if bbox:
        lon = (bbox[0] + bbox[2]) / 2
        lat = (bbox[1] + bbox[3]) / 2
    else:
        lat, lon = 44.4939, 11.3426  # Bologna centro

    m = folium.Map(location=[lat, lon], zoom_start=zoom_start, tiles=None, control_scale=True)

    url = f"http://sitmappe.comune.bologna.it/tms/tileserver/Ortofoto{year}/{{z}}/{{x}}/{{y}}.png"
    # If tiles appear vertically flipped, set tms=True below.
    folium.TileLayer(
        tiles=url,
        name=f"Ortofoto {year}",
        attr="© Comune di Bologna – Ortofoto",
        overlay=True,
        control=True,
        max_zoom=20,
        # tms=False  # default; switch to True only if you see a vertical flip
    ).add_to(m)

    if bbox:
        m.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])

    if save_path:
        m.save(save_path)
        return save_path  # could also return None if you prefer
    else:
        return m

@tool(
    name_or_callable="get_ortofoto",
    description="Download ortofoto for a given year, centered around a specific location (if provided) from Comune di Bologna.",
)
# bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
async def folium_ortho(
    year: Annotated[int, "year of reference"], 
    runtime: ToolRuntime,
    query: Optional[Annotated[str, "query to get the ortofoto of a specific location"]]=None,
    )-> Command:

    bbox = None
    # if the agent queried a location, build bbox taking the location as the center
    if query:
        try:
            result = get_google_geocoding_tool().run(query)
            # coordinates of the center of the bbox
            lat, lon = result[0]['geometry']['location']['lat'], result[0]['geometry']['location']['lng']
            # construct bbox from lat and lon in a square of side 1000 meters
            bbox = [lon - 0.0025, lat - 0.0025, lon + 0.0025, lat + 0.0025]
        except Exception as e:
            print(f"Error getting coordinates of the location: {e}")
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error getting coordinates of the location: {e}",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

    try:
        # Generate the folium map in a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
        
        folium_map = folium_ortho_fn(
            year=year,
            bbox=bbox,
            zoom_start=16,
            save_path=temp_path
        )
        
        # Get session and container
        session_id = get_session_key()
        session_manager.start(session_id)
        
        # Ingest the file directly into the artifact system
        from backend.artifacts.ingest import ingest_files
        from backend.graph.context import get_db_session, get_thread_id
        
        db_session = get_db_session()
        thread_id = get_thread_id()
        
        descriptors = await ingest_files(
            session=db_session,
            thread_id=thread_id,
            new_host_files=[temp_path],
            session_id=session_id,
            run_id=f"ortofoto_{year}",
            tool_call_id=runtime.tool_call_id
        )
        
        # ingest_files automatically deletes the temp file, so no cleanup needed
        
        if descriptors:
            artifact = descriptors[0]
            structured_artifact = {
                "name": artifact.get('name', f'ortofoto_{year}.html'),
                "mime": artifact.get('mime', 'text/html'),
                "url": artifact.get('url', ''),
                "size": artifact.get('size', 0)
            }
            
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Ortofoto {year} map generated successfully",
                            artifact=[structured_artifact],
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )
        else:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Failed to process ortofoto {year} map",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )
        
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error generating ortofoto {year}: {str(e)}",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )

# -------------------------------
# compare_ortofoto tool (DualMap)
# -------------------------------

# Import your session helpers
from backend.graph.dataset_tools import session_manager, get_session_key

_AVAILABLE_YEARS = {2017, 2018, 2020, 2021, 2022, 2023, 2024}

def _make_dual_ortho_map(
    left_year: int,
    right_year: int,
    bbox: Optional[List[float]] = None,
    zoom_start: int = 16,
    tms: bool = False,
    divider_px: int = 4,
):
    """
    Build two synchronized Leaflet maps (left/right) showing the same extent
    with different ortofoto years. bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84.
    """
    note_parts = []
    if left_year not in _AVAILABLE_YEARS or right_year not in _AVAILABLE_YEARS:
        note_parts.append(
            f"Requested year not available. Valid years: {sorted(_AVAILABLE_YEARS)}."
        )

    if left_year == right_year:
        note_parts.append(
            f"Warning: left_year == right_year == {left_year}. Panels will be identical."
        )

    # Default center: Bologna centro
    center_lat, center_lon = 44.4939, 11.3426
    min_lon = min_lat = max_lon = max_lat = None

    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        if max_lon < min_lon:
            min_lon, max_lon = max_lon, min_lon
        if max_lat < min_lat:
            min_lat, max_lat = max_lat, min_lat
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2

    m = DualMap(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )

    def _tile(year: int, side_label: str) -> folium.TileLayer:
        # Use HTTPS to avoid mixed-content issues
        url = f"https://sitmappe.comune.bologna.it/tms/tileserver/Ortofoto{year}/{{z}}/{{x}}/{{y}}.png"
        return folium.TileLayer(
            tiles=url,
            name=f"Ortofoto {year}",
            attr=f"© Comune di Bologna - {side_label} {year}",
            overlay=False,         # base layer in each panel
            control=True,
            max_zoom=20,
            max_native_zoom=20,
            tms=tms,
        )

    _tile(left_year,  "LEFT").add_to(m.m1)
    _tile(right_year, "RIGHT").add_to(m.m2)

    # Optional LayerControl per panel
    folium.LayerControl(collapsed=True).add_to(m.m1)
    folium.LayerControl(collapsed=True).add_to(m.m2)

    # Keep both panels on the same extent
    if bbox and None not in (min_lat, min_lon, max_lat, max_lon):
        m.m1.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])   # type: ignore
        m.m2.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])   # type: ignore

    # Visible vertical divider via CSS borders on both panel containers
    left_id  = m.m1.get_name()
    right_id = m.m2.get_name()
    css = f"""
    <style>
      #{left_id}  {{ border-right: {divider_px}px solid rgba(255,255,255,0.9) !important; }}
      #{right_id} {{ border-left:  {divider_px}px solid rgba(255,255,255,0.9) !important; }}
    </style>
    """
    m.get_root().html.add_child(Element(css)) # type: ignore

    note = " ".join(note_parts) if note_parts else None
    return m, note

@tool(
    name_or_callable="compare_ortofoto",
    description=(
        "Generate a single HTML showing two synchronized panels (left/right) with Bologna ortofoto "
        "for two years over the same area. Optional query to center the ortofoto around a specific location."
    ),
)
async def compare_ortofoto(
    left_year: Annotated[int, "left ortofoto year"],
    right_year: Annotated[int, "right ortofoto year"],
    query: Optional[Annotated[str, "name of the location to center the ortofoto around"]]=None,
    runtime: ToolRuntime = None,
) -> Command:
    """
    Returns a ToolMessage with:
      - an artifact (HTML file) for download
      - inline HTML (additional_kwargs['html']) for in-UI rendering
    """

    bbox = None
    # if the agent queried a location, build bbox taking the location as the center
    if query:
        try:
            result = get_google_geocoding_tool().run(query)
            # coordinates of the center of the bbox
            lat, lon = result[0]['geometry']['location']['lat'], result[0]['geometry']['location']['lng']
            # construct bbox from lat and lon in a square of side 1000 meters
            bbox = [lon - 0.0025, lat - 0.0025, lon + 0.0025, lat + 0.0025] # these 0.005 are degrees, not meters <- estimate
        except Exception as e:
            print(f"Error getting coordinates of the location: {e}")
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error getting coordinates of the location: {e}",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

    try:
        m, note = _make_dual_ortho_map(
            left_year=left_year,
            right_year=right_year,
            bbox=bbox,
            zoom_start=16,
            tms=False,
        )

        # Save to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
            temp_path = Path(tmp.name)
        m.save(temp_path)

        # Read inline HTML for Chainlit CustomElement
        html_content = temp_path.read_text(encoding="utf-8")

        # Start / acquire session
        session_id = get_session_key()
        session_manager.start(session_id)

        # Ingest artifact
        from backend.artifacts.ingest import ingest_files
        from backend.graph.context import get_db_session, get_thread_id
        
        db_session = get_db_session()
        thread_id = get_thread_id()
        
        descriptors = await ingest_files(
            session=db_session,
            thread_id=thread_id,
            new_host_files=[temp_path],
            session_id=session_id,
            run_id=f"compare_ortofoto_dual_{left_year}_vs_{right_year}",
            tool_call_id=runtime.tool_call_id,
        )
        # ingest_files auto-deletes temp file

        artifact_struct = None
        if descriptors: # means it was ingested correctly

            message = f"Dual ortofoto: {left_year} (left) vs {right_year} (right) generated and shown successfully"
            if note:
                message += f" {note}"

            a = descriptors[0]
            artifact_struct = {
                "name": a.get("name", f"ortofoto_dual_{left_year}_vs_{right_year}.html"),
                "mime": a.get("mime", "text/html"),
                "url": a.get("url", ""),
                "size": a.get("size", 0),
            }
            
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=message,
                            artifact=[artifact_struct],
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )
        else: # means it was not ingested correctly

            message = f"Failed to ingest and show ortofoto {left_year} (left) vs {right_year} (right) map"
            if note:
                message += f" {note}"

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=message,
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )
        
    except Exception as e: # means it was not generated correctly
        return Command(
            update={    
                "messages": [
                    ToolMessage(
                        content=f"Error generating ortofoto {left_year} (left) vs {right_year} (right) map: {str(e)}",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )

# -------------------------------
# view_3d_model tool
# -------------------------------

def _create_bologna_3d_html() -> str:
    """
    Create HTML content for the Bologna 3D model viewer.
    Returns a minimal iframe embedding (fast and lightweight).
    """
    # Metodo 2 dal notebook: iframe minimal con styling essenziale
    html_content = """<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bologna 3D</title>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; }
        .viewer-container {
            width: 100%;
            height: 100vh;
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        iframe {
            width: 100%;
            height: 100%;
            border: none;
            display: block;
        }
    </style>
</head>
<body>
    <div class="viewer-container">
        <iframe 
            src="https://sitmappe.comune.bologna.it/Bologna3D/" 
            allowfullscreen
            title="Bologna 3D Model">
        </iframe>
    </div>
</body>
</html>"""
    return html_content


@tool(
    name_or_callable="view_3d_model",
    description=(
        "Display the interactive 3D model of Bologna city. "
        "Use this tool to show a three-dimensional visualization of the city buildings and terrain. "
        "The model is interactive: users can rotate, zoom, and navigate through the 3D representation."
    ),
)
async def view_3d_model(
    runtime: ToolRuntime,
) -> Command:
    """
    Generate an interactive 3D model viewer for Bologna.
    """
    try:
        # Generate the HTML content
        html_content = _create_bologna_3d_html()
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
            tmp.write(html_content)
            temp_path = Path(tmp.name)
        
        # Get session and container
        session_id = get_session_key()
        session_manager.start(session_id)
        
        # Ingest the file into the artifact system
        from backend.artifacts.ingest import ingest_files
        from backend.graph.context import get_db_session, get_thread_id
        
        db_session = get_db_session()
        thread_id = get_thread_id()
        
        descriptors = await ingest_files(
            session=db_session,
            thread_id=thread_id,
            new_host_files=[temp_path],
            session_id=session_id,
            run_id="bologna_3d_model",
            tool_call_id=runtime.tool_call_id
        )
        # ingest_files automatically deletes the temp file
        
        if descriptors:
            artifact = descriptors[0]
            artifact_struct = {
                "name": artifact.get("name", "bologna_3d_model.html"),
                "mime": artifact.get("mime", "text/html"),
                "url": artifact.get("url", ""),
                "size": artifact.get("size", 0),
            }
            
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Bologna 3D model viewer generated successfully. You can now explore the city in 3D.",
                            artifact=[artifact_struct],
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )
        else:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Failed to generate the 3D model viewer",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )
            
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error generating 3D model viewer: {str(e)}",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )
#!/usr/bin/env python3
"""
DJ Lights MCP Server - Control Kasa smart bulbs via MCP tools.

Exposes all DJ Lights capabilities (visualization modes, color control,
status queries) as MCP tools for use with Claude or any MCP client.

Usage:
    python dj_lights_mcp.py                  # stdio transport (for Claude Code / Desktop)
    mcp dev dj_lights_mcp.py                 # MCP Inspector for testing

Environment Variables:
    BULBS - Comma or colon separated list of bulb IPs
            Format: "name1:ip1,name2:ip2" or just "ip1,ip2"
            Example: "left:192.168.0.103,right:192.168.0.104"
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import sounddevice as sd
from mcp.server.fastmcp import FastMCP

from dj_lights import (
    AlbumReactiveMapper,
    AudioAnalyzer,
    BulbController,
    CalmMapper,
    FixedHueMapper,
    MovieMapper,
    RaveMapper,
    ReactiveMapper,
    extract_dominant_colors,
    get_album_artwork,
    get_current_track_info,
    list_app_windows,
    parse_bulbs_env,
    run_movie_visualization,
    run_visualization,
    run_white_mode,
)


@dataclass
class AppState:
    """Shared state for the MCP server."""

    controller: BulbController | None = None
    current_task: asyncio.Task | None = None
    stop_event: asyncio.Event | None = None
    current_mode: str = "idle"
    current_settings: dict = field(default_factory=dict)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    """Connect to bulbs on startup, disconnect on shutdown."""
    state = AppState()

    bulb_ips = parse_bulbs_env()
    if bulb_ips:
        state.controller = BulbController(bulb_ips)
        try:
            await state.controller.connect()
            await state.controller.turn_on_all()
        except Exception as e:
            print(f"Warning: Failed to connect to bulbs: {e}")
            state.controller = None

    try:
        yield state
    finally:
        # Cleanup
        if state.current_task and not state.current_task.done():
            if state.stop_event:
                state.stop_event.set()
            await asyncio.sleep(0.3)
        if state.controller:
            await state.controller.disconnect()


mcp = FastMCP(
    "DJ Lights",
    instructions=(
        "Control TP-Link Kasa smart bulbs with real-time music visualization. "
        "Use set_mode to start visualizations, set_color for manual colors, "
        "and get_status to check the current state."
    ),
    lifespan=lifespan,
)


def _require_controller(state: AppState) -> BulbController:
    """Raise if no controller is connected."""
    if state.controller is None:
        raise ValueError(
            "No bulbs connected. Set the BULBS environment variable "
            "(e.g. BULBS='left:192.168.0.103,right:192.168.0.104') and restart the server."
        )
    return state.controller


async def _stop_current(state: AppState):
    """Stop any running visualization."""
    if state.current_task and not state.current_task.done():
        if state.stop_event:
            state.stop_event.set()
        await asyncio.sleep(0.3)
    state.current_task = None
    state.stop_event = None
    state.current_mode = "idle"


# ── Light Control Tools ───────────────────────────────────────────────


@mcp.tool()
async def set_mode(
    mode: str,
    device: int | None = None,
    sensitivity: float = 1.0,
    color_mode: str = "sync",
    max_brightness: int = 100,
    hue: int = 240,
    window: str = "Google Chrome",
    window_title: str | None = None,
    audio_blend: float = 0.3,
    color_temp: str = "warm1",
    brightness: int = 80,
) -> str:
    """Start a visualization mode on the smart bulbs.

    Args:
        mode: Visualization mode - one of: reactive, album, calm, rave, fixed, movie, white.
              reactive = full spectrum beat-driven colors.
              album = colors extracted from current Apple Music album artwork.
              calm = cool blues/greens, smooth transitions (good for studying).
              rave = fast cycling, strobe on beats, max saturation.
              fixed = single hue, music controls brightness only.
              movie = screen capture ambilight-style (left/right bulb colors from screen halves).
              white = static white light at configurable color temperature.
        device: Audio input device index (use list_audio_devices to find it). None for default.
        sensitivity: Beat detection sensitivity (0.5=less, 2.0=more, default=1.0).
        color_mode: 'sync' (all bulbs same color) or 'complementary' (opposite colors on each bulb).
        max_brightness: Maximum brightness cap 1-100 (default 100).
        hue: Fixed hue value 0-360 for 'fixed' mode (default 240=blue).
        window: Target app name for 'movie' mode (default 'Google Chrome').
        window_title: Window title to match for 'movie' mode (partial match).
        audio_blend: Audio influence on movie mode brightness 0-1 (default 0.3).
        color_temp: Color temperature for 'white' mode: cool2, cool1, neutral, warm1, warm2.
        brightness: Brightness for 'white' mode 1-100 (default 80).
    """
    ctx = mcp.get_context()
    state: AppState = ctx.request_context.lifespan_context
    controller = _require_controller(state)

    valid_modes = ["reactive", "album", "calm", "rave", "fixed", "movie", "white"]
    if mode not in valid_modes:
        return json.dumps({"error": f"Invalid mode '{mode}'. Choose from: {valid_modes}"})

    # Stop any running visualization
    await _stop_current(state)

    sensitivity = max(0.1, min(3.0, sensitivity))
    max_brightness = max(1, min(100, max_brightness))

    # White mode — no audio needed
    if mode == "white":
        await run_white_mode(controller, brightness, color_temp, disconnect=False)
        state.current_mode = "white"
        state.current_settings = {"color_temp": color_temp, "brightness": brightness}
        return json.dumps({"status": "ok", "mode": "white", "color_temp": color_temp, "brightness": brightness})

    # Movie mode — screen capture
    if mode == "movie":
        audio_blend = max(0.0, min(1.0, audio_blend))
        mapper = MovieMapper(
            sensitivity=sensitivity,
            audio_blend=audio_blend,
            window_name=window,
            window_title=window_title or window,
            max_brightness=max_brightness,
        )
        analyzer = AudioAnalyzer(device=device) if audio_blend > 0 else None
        state.stop_event = asyncio.Event()
        state.current_task = asyncio.create_task(
            run_movie_visualization(controller, mapper, analyzer, state.stop_event)
        )
        state.current_mode = "movie"
        state.current_settings = {
            "window": window,
            "window_title": window_title,
            "audio_blend": audio_blend,
            "sensitivity": sensitivity,
            "max_brightness": max_brightness,
        }
        return json.dumps({"status": "ok", "mode": "movie", **state.current_settings})

    # Audio visualization modes
    mappers = {
        "reactive": lambda: ReactiveMapper(sensitivity=sensitivity, max_brightness=max_brightness),
        "album": lambda: AlbumReactiveMapper(sensitivity=sensitivity, max_brightness=max_brightness),
        "calm": lambda: CalmMapper(sensitivity=sensitivity, max_brightness=max_brightness),
        "rave": lambda: RaveMapper(sensitivity=sensitivity, max_brightness=max_brightness),
        "fixed": lambda: FixedHueMapper(hue=hue % 360, sensitivity=sensitivity, max_brightness=max_brightness),
    }
    mapper = mappers[mode]()
    analyzer = AudioAnalyzer(device=device)

    state.stop_event = asyncio.Event()
    state.current_task = asyncio.create_task(
        run_visualization(controller, analyzer, mapper, color_mode, state.stop_event)
    )
    state.current_mode = mode
    state.current_settings = {
        "color_mode": color_mode,
        "sensitivity": sensitivity,
        "max_brightness": max_brightness,
        "device": device,
    }
    if mode == "fixed":
        state.current_settings["hue"] = hue % 360

    return json.dumps({"status": "ok", "mode": mode, **state.current_settings})


@mcp.tool()
async def set_color(
    hue: int,
    saturation: int = 100,
    brightness: int = 100,
    bulb_name: str | None = None,
) -> str:
    """Set a specific HSV color on the bulbs. Stops any running visualization first.

    Args:
        hue: Hue value 0-360 (0=red, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta).
        saturation: Saturation 0-100 (default 100).
        brightness: Brightness 0-100 (default 100).
        bulb_name: Optional bulb name to target. If omitted, sets all bulbs.
    """
    ctx = mcp.get_context()
    state: AppState = ctx.request_context.lifespan_context
    controller = _require_controller(state)

    await _stop_current(state)

    hue = hue % 360
    saturation = max(0, min(100, saturation))
    brightness = max(0, min(100, brightness))

    if bulb_name:
        if bulb_name not in controller.bulbs:
            names = list(controller.bulbs.keys())
            return json.dumps({"error": f"Bulb '{bulb_name}' not found. Available: {names}"})
        await controller.set_hsv(bulb_name, hue, saturation, brightness)
    else:
        await controller.set_all_hsv(hue, saturation, brightness)

    state.current_mode = "manual"
    return json.dumps({
        "status": "ok",
        "hue": hue,
        "saturation": saturation,
        "brightness": brightness,
        "bulb": bulb_name or "all",
    })


@mcp.tool()
async def turn_on() -> str:
    """Turn on all connected bulbs."""
    ctx = mcp.get_context()
    state: AppState = ctx.request_context.lifespan_context
    controller = _require_controller(state)
    await controller.turn_on_all()
    return json.dumps({"status": "ok", "action": "turned_on"})


@mcp.tool()
async def turn_off() -> str:
    """Turn off all connected bulbs. Stops any running visualization first."""
    ctx = mcp.get_context()
    state: AppState = ctx.request_context.lifespan_context
    controller = _require_controller(state)
    await _stop_current(state)
    await controller.turn_off_all()
    state.current_mode = "off"
    return json.dumps({"status": "ok", "action": "turned_off"})


@mcp.tool()
async def stop() -> str:
    """Stop the current visualization. Bulbs remain on at their last color."""
    ctx = mcp.get_context()
    state: AppState = ctx.request_context.lifespan_context
    _require_controller(state)

    was_running = state.current_task and not state.current_task.done()
    previous_mode = state.current_mode
    await _stop_current(state)

    if was_running:
        return json.dumps({"status": "ok", "stopped": previous_mode})
    return json.dumps({"status": "ok", "message": "No visualization was running"})


@mcp.tool()
async def test_colors() -> str:
    """Cycle through test colors on all bulbs (one color per second, 12 steps)."""
    ctx = mcp.get_context()
    state: AppState = ctx.request_context.lifespan_context
    controller = _require_controller(state)

    await _stop_current(state)

    for hue in range(0, 360, 30):
        await controller.set_all_hsv(hue, 100, 100)
        await asyncio.sleep(1)

    state.current_mode = "idle"
    return json.dumps({"status": "ok", "action": "test_complete"})


# ── Info / Status Tools ───────────────────────────────────────────────


@mcp.tool()
async def get_status() -> str:
    """Get current DJ Lights status: mode, connected bulbs, visualization state, and current track."""
    ctx = mcp.get_context()
    state: AppState = ctx.request_context.lifespan_context

    result = {
        "mode": state.current_mode,
        "visualization_running": state.current_task is not None and not state.current_task.done(),
        "settings": state.current_settings,
        "connected_bulbs": [],
    }

    if state.controller:
        for name, bulb in state.controller.bulbs.items():
            last_hsv = state.controller.last_hsv.get(name, (0, 0, 0))
            result["connected_bulbs"].append({
                "name": name,
                "ip": state.controller.ips.get(name, "unknown"),
                "last_hsv": {"hue": last_hsv[0], "saturation": last_hsv[1], "brightness": last_hsv[2]},
            })

    # Try to get current track
    track_info = get_current_track_info()
    if track_info:
        result["current_track"] = track_info

    return json.dumps(result)


@mcp.tool()
async def list_audio_devices() -> str:
    """List available audio input devices with their indices. Use the index with set_mode's device parameter."""
    devices = sd.query_devices()
    inputs = []
    default_input = sd.default.device[0]

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            inputs.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "is_default": i == default_input,
            })

    return json.dumps({"devices": inputs, "tip": "Look for 'BlackHole' or a virtual audio device for system audio capture."})


@mcp.tool()
async def get_track_info() -> str:
    """Get the currently playing track from Apple Music."""
    track_info = get_current_track_info()
    if track_info:
        return json.dumps({"playing": True, **track_info})
    return json.dumps({"playing": False, "message": "No track playing or Apple Music not running."})


@mcp.tool()
async def list_bulbs() -> str:
    """List all configured and connected bulbs."""
    ctx = mcp.get_context()
    state: AppState = ctx.request_context.lifespan_context

    configured = parse_bulbs_env()
    connected = list(state.controller.bulbs.keys()) if state.controller else []

    bulbs = []
    for name, ip in configured.items():
        bulbs.append({
            "name": name,
            "ip": ip,
            "connected": name in connected,
        })

    return json.dumps({"bulbs": bulbs})


@mcp.tool()
async def list_windows(app_name: str = "Google Chrome") -> str:
    """List visible windows for an application. Useful for movie mode setup.

    Args:
        app_name: Application name to list windows for (default 'Google Chrome').
    """
    windows = list_app_windows(app_name)
    if not windows:
        return json.dumps({"windows": [], "message": f"No windows found for '{app_name}'. Make sure the app is open."})

    result = []
    for i, win in enumerate(sorted(windows, key=lambda w: w["width"] * w["height"], reverse=True)):
        result.append({
            "index": i,
            "title": win["title"],
            "width": win["width"],
            "height": win["height"],
        })

    return json.dumps({"app": app_name, "windows": result})


if __name__ == "__main__":
    mcp.run(transport="stdio")

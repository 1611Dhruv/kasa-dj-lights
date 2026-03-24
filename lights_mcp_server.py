"""DJ Lights MCP Server — gives Claude direct control over Kasa smart lights."""

import asyncio
import json
import sys
import threading

from kasa import Discover
from kasa.module import Module
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "dj-lights",
    instructions=(
        "You control Kasa smart lights on the local network. "
        "Use discover_lights first to find bulbs, then control them with "
        "set_color, set_scene, turn_on/off, or start music visualizations. "
        "When the user says 'vibe out' or 'set the mood', pick a scene that "
        "matches. Pair with the apple-music MCP to create full DJ experiences."
    ),
)

# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------

_bulbs: dict[str, object] = {}  # name -> kasa device
_viz_stop_event: threading.Event | None = None
_viz_thread: threading.Thread | None = None
_viz_mode: str | None = None

# ---------------------------------------------------------------------------
# Scenes — predefined HSV palettes (hue, saturation, brightness)
# ---------------------------------------------------------------------------

SCENES = {
    "party": {
        "description": "Vibrant cycling colors — great for dancing",
        "colors": [(0, 100, 100), (60, 100, 100), (120, 100, 100), (240, 100, 100), (300, 100, 100)],
        "cycle": True,
        "speed": 1.0,
    },
    "chill": {
        "description": "Warm amber and soft orange — relaxed evening vibes",
        "colors": [(30, 80, 60), (40, 70, 50)],
        "cycle": False,
    },
    "movie": {
        "description": "Dim warm backlight — minimal distraction for watching",
        "colors": [(30, 40, 25)],
        "cycle": False,
    },
    "sunset": {
        "description": "Orange to deep pink gradient",
        "colors": [(15, 90, 80), (340, 80, 70)],
        "cycle": False,
    },
    "ocean": {
        "description": "Cool blues and teals — calm water feel",
        "colors": [(200, 80, 60), (180, 70, 50)],
        "cycle": False,
    },
    "forest": {
        "description": "Greens and earthy tones — nature vibes",
        "colors": [(120, 70, 50), (90, 60, 40)],
        "cycle": False,
    },
    "romantic": {
        "description": "Soft pinks and warm reds — date night",
        "colors": [(340, 60, 50), (350, 50, 40)],
        "cycle": False,
    },
    "focus": {
        "description": "Cool white with slight blue — sharp concentration",
        "colors": [(210, 20, 80)],
        "cycle": False,
    },
    "gaming": {
        "description": "Purple and cyan — RGB gamer aesthetic",
        "colors": [(270, 100, 80), (180, 100, 80)],
        "cycle": False,
    },
    "deep_purple": {
        "description": "Rich purples — moody and atmospheric",
        "colors": [(270, 80, 60), (290, 70, 50)],
        "cycle": False,
    },
    "lava": {
        "description": "Deep reds and oranges — volcanic energy",
        "colors": [(0, 100, 80), (20, 90, 70)],
        "cycle": False,
    },
    "aurora": {
        "description": "Greens and purples — northern lights",
        "colors": [(130, 80, 60), (280, 70, 60)],
        "cycle": False,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(msg: str) -> str:
    return json.dumps({"error": msg})


def _ok(data: dict) -> str:
    return json.dumps(data)


def _get_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop for async kasa operations."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def _run_async(coro):
    """Run an async coroutine from sync context."""
    loop = _get_loop()
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@mcp.tool()
def discover_lights(timeout: int = 5) -> str:
    """Discover all Kasa smart lights on the local network. Run this first before controlling lights. Timeout in seconds (default 5)."""
    global _bulbs

    async def _discover():
        devices = await Discover.discover(timeout=timeout)
        found = {}
        for ip, dev in devices.items():
            await dev.update()
            if Module.Light in dev.modules:
                name = dev.alias or str(ip)
                found[name] = dev
        return found

    try:
        found = _run_async(_discover())
        _bulbs.update(found)

        lights = []
        for name, dev in _bulbs.items():
            light = dev.modules[Module.Light]
            hsv = light.hsv
            lights.append({
                "name": name,
                "ip": str(dev.host),
                "is_on": dev.is_on,
                "hue": hsv.hue if hasattr(hsv, 'hue') else hsv[0],
                "saturation": hsv.saturation if hasattr(hsv, 'saturation') else hsv[1],
                "brightness": hsv.brightness if hasattr(hsv, 'brightness') else hsv[2],
            })

        return _ok({"discovered": len(found), "total": len(_bulbs), "lights": lights})
    except Exception as e:
        return _error(f"Discovery failed: {e}")


@mcp.tool()
def list_lights() -> str:
    """List all currently known lights and their state. Run discover_lights first if empty."""
    if not _bulbs:
        return _ok({"lights": [], "hint": "No lights known yet. Run discover_lights first."})

    async def _update_all():
        for dev in _bulbs.values():
            try:
                await dev.update()
            except Exception:
                pass

    try:
        _run_async(_update_all())
    except Exception:
        pass

    lights = []
    for name, dev in _bulbs.items():
        info = {"name": name, "ip": str(dev.host), "is_on": dev.is_on}
        if Module.Light in dev.modules:
            light = dev.modules[Module.Light]
            hsv = light.hsv
            info["hue"] = hsv.hue if hasattr(hsv, 'hue') else hsv[0]
            info["saturation"] = hsv.saturation if hasattr(hsv, 'saturation') else hsv[1]
            info["brightness"] = hsv.brightness if hasattr(hsv, 'brightness') else hsv[2]
        lights.append(info)

    return _ok({"lights": lights})


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

@mcp.tool()
def turn_on(name: str = "") -> str:
    """Turn on a light by name, or all lights if name is empty."""
    if not _bulbs:
        return _error("No lights known. Run discover_lights first.")

    async def _on():
        targets = [_bulbs[name]] if name and name in _bulbs else list(_bulbs.values())
        for dev in targets:
            await dev.turn_on()
            await dev.update()
        return len(targets)

    try:
        count = _run_async(_on())
        return _ok({"turned_on": count})
    except Exception as e:
        return _error(str(e))


@mcp.tool()
def turn_off(name: str = "") -> str:
    """Turn off a light by name, or all lights if name is empty."""
    if not _bulbs:
        return _error("No lights known. Run discover_lights first.")

    async def _off():
        targets = [_bulbs[name]] if name and name in _bulbs else list(_bulbs.values())
        for dev in targets:
            await dev.turn_off()
            await dev.update()
        return len(targets)

    try:
        count = _run_async(_off())
        return _ok({"turned_off": count})
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# Color Control
# ---------------------------------------------------------------------------

@mcp.tool()
def set_color(hue: int, saturation: int = 100, brightness: int = 80, name: str = "") -> str:
    """Set a light to an HSV color. Hue 0-360 (0=red, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta). Saturation 0-100. Brightness 0-100. If name is empty, sets all lights."""
    if not _bulbs:
        return _error("No lights known. Run discover_lights first.")

    hue = hue % 360
    saturation = max(0, min(100, saturation))
    brightness = max(0, min(100, brightness))

    async def _set():
        targets = [_bulbs[name]] if name and name in _bulbs else list(_bulbs.values())
        for dev in targets:
            if not dev.is_on:
                await dev.turn_on()
            light = dev.modules[Module.Light]
            await light.set_hsv(hue, saturation, brightness)
        return len(targets)

    try:
        count = _run_async(_set())
        return _ok({"set_color": count, "hue": hue, "saturation": saturation, "brightness": brightness})
    except Exception as e:
        return _error(str(e))


@mcp.tool()
def set_white(brightness: int = 80, temperature: str = "neutral", name: str = "") -> str:
    """Set white light. Temperature: cool2 (daylight), cool1, neutral, warm1, warm2 (candlelight). Brightness 0-100."""
    if not _bulbs:
        return _error("No lights known. Run discover_lights first.")

    temps = {
        "cool2": (210, 25),
        "cool1": (180, 20),
        "neutral": (60, 15),
        "warm1": (35, 30),
        "warm2": (25, 40),
    }
    if temperature not in temps:
        return _error(f"Unknown temperature '{temperature}'. Use: {', '.join(temps)}")

    hue, sat = temps[temperature]
    brightness = max(0, min(100, brightness))

    async def _set():
        targets = [_bulbs[name]] if name and name in _bulbs else list(_bulbs.values())
        for dev in targets:
            if not dev.is_on:
                await dev.turn_on()
            light = dev.modules[Module.Light]
            # Try native color_temp if supported
            if light.has_feature("color_temp"):
                feat = light.get_feature("color_temp")
                lo, hi = feat.minimum_value, feat.maximum_value
                rng = hi - lo
                temp_map = {"cool2": hi, "cool1": lo + int(rng * 0.75), "neutral": lo + int(rng * 0.5), "warm1": lo + int(rng * 0.25), "warm2": lo}
                await light.set_color_temp(temp_map[temperature])
                await light.set_brightness(brightness)
            else:
                await light.set_hsv(hue, sat, brightness)
        return len(targets)

    try:
        count = _run_async(_set())
        return _ok({"set_white": count, "temperature": temperature, "brightness": brightness})
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

@mcp.tool()
def set_scene(scene: str) -> str:
    """Apply a predefined lighting scene. Available scenes: party, chill, movie, sunset, ocean, forest, romantic, focus, gaming, deep_purple, lava, aurora. Each sets mood-appropriate colors on your lights."""
    if not _bulbs:
        return _error("No lights known. Run discover_lights first.")

    if scene not in SCENES:
        available = {k: v["description"] for k, v in SCENES.items()}
        return _error(f"Unknown scene '{scene}'. Available: {json.dumps(available)}")

    scene_data = SCENES[scene]
    colors = scene_data["colors"]

    async def _apply():
        bulb_list = list(_bulbs.values())
        for i, dev in enumerate(bulb_list):
            if not dev.is_on:
                await dev.turn_on()
            color = colors[i % len(colors)]
            light = dev.modules[Module.Light]
            await light.set_hsv(color[0], color[1], color[2])
        return len(bulb_list)

    try:
        count = _run_async(_apply())
        return _ok({
            "scene": scene,
            "description": scene_data["description"],
            "lights_set": count,
        })
    except Exception as e:
        return _error(str(e))


@mcp.tool()
def list_scenes() -> str:
    """List all available lighting scenes with descriptions."""
    scenes = {name: data["description"] for name, data in SCENES.items()}
    return _ok({"scenes": scenes})


# ---------------------------------------------------------------------------
# Visualization (runs audio-reactive light shows in background thread)
# ---------------------------------------------------------------------------

def _run_visualization_thread(mode: str, device, sensitivity: float, color_mode: str, stop_event: threading.Event):
    """Run a music visualization in a background thread."""
    # Import here to avoid loading heavy deps unless needed
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from dj_lights import (
        AudioAnalyzer,
        BulbController,
        ReactiveMapper,
        AlbumReactiveMapper,
        CalmMapper,
        RaveMapper,
        FixedHueMapper,
    )

    async def _viz():
        # Build a BulbController from our already-discovered bulbs
        controller = BulbController({})
        controller.bulbs = dict(_bulbs)
        controller.last_hsv = {n: (-999, -999, -999) for n in _bulbs}

        mappers = {
            "reactive": lambda: ReactiveMapper(sensitivity=sensitivity),
            "album": lambda: AlbumReactiveMapper(sensitivity=sensitivity),
            "calm": lambda: CalmMapper(sensitivity=sensitivity),
            "rave": lambda: RaveMapper(sensitivity=sensitivity),
        }
        mapper = mappers.get(mode, mappers["reactive"])()
        analyzer = AudioAnalyzer(device=device)

        # Use asyncio stop event
        astop = asyncio.Event()

        # Poll threading event in background
        async def _poll_stop():
            while not stop_event.is_set():
                await asyncio.sleep(0.2)
            astop.set()

        poll_task = asyncio.create_task(_poll_stop())

        from dj_lights import run_visualization
        await run_visualization(controller, analyzer, mapper, color_mode, stop_event=astop)
        poll_task.cancel()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_viz())
    except Exception as e:
        print(f"Visualization error: {e}", file=sys.stderr)
    finally:
        loop.close()


@mcp.tool()
def start_visualization(mode: str = "reactive", device: int = -1, sensitivity: float = 1.0, color_mode: str = "sync") -> str:
    """Start a music-reactive light visualization. Mode: reactive (full spectrum beats), album (artwork colors), calm (blues/greens), rave (strobe party). Device: audio input device index (-1 for default). Sensitivity: 0.5-3.0 (default 1.0). Color mode: sync (all same) or complementary (opposite colors)."""
    global _viz_stop_event, _viz_thread, _viz_mode

    if not _bulbs:
        return _error("No lights known. Run discover_lights first.")

    valid_modes = ["reactive", "album", "calm", "rave"]
    if mode not in valid_modes:
        return _error(f"Unknown mode '{mode}'. Use: {', '.join(valid_modes)}")

    # Stop existing visualization
    if _viz_thread and _viz_thread.is_alive():
        _viz_stop_event.set()
        _viz_thread.join(timeout=3)

    dev = None if device < 0 else device
    _viz_stop_event = threading.Event()
    _viz_thread = threading.Thread(
        target=_run_visualization_thread,
        args=(mode, dev, sensitivity, color_mode, _viz_stop_event),
        daemon=True,
    )
    _viz_thread.start()
    _viz_mode = mode

    return _ok({"status": "started", "mode": mode, "sensitivity": sensitivity, "color_mode": color_mode})


@mcp.tool()
def stop_visualization() -> str:
    """Stop the currently running music visualization."""
    global _viz_stop_event, _viz_thread, _viz_mode

    if not _viz_thread or not _viz_thread.is_alive():
        return _ok({"status": "no_visualization_running"})

    _viz_stop_event.set()
    _viz_thread.join(timeout=5)
    mode = _viz_mode
    _viz_mode = None

    return _ok({"status": "stopped", "was_running": mode})


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@mcp.tool()
def get_status() -> str:
    """Get overall status: connected lights, running visualization, etc."""
    viz_running = _viz_thread is not None and _viz_thread.is_alive()

    return _ok({
        "lights_count": len(_bulbs),
        "light_names": list(_bulbs.keys()),
        "visualization_running": viz_running,
        "visualization_mode": _viz_mode if viz_running else None,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")

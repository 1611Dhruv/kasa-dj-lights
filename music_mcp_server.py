"""Apple Music MCP Server — gives Claude full control over Apple Music on macOS."""

import json
import ssl
import subprocess
import urllib.parse
import urllib.request

import certifi
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("apple-music")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_applescript(script: str) -> tuple[bool, str]:
    """Run an AppleScript and return (success, output)."""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, result.stdout.strip()


def error_response(msg: str) -> str:
    return json.dumps({"error": msg})


def success_response(data: dict) -> str:
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Playback Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def play() -> str:
    """Resume playback in Apple Music."""
    ok, out = run_applescript('tell application "Music" to play')
    if not ok:
        return error_response(out)
    return success_response({"status": "playing"})


@mcp.tool()
def pause() -> str:
    """Pause playback in Apple Music."""
    ok, out = run_applescript('tell application "Music" to pause')
    if not ok:
        return error_response(out)
    return success_response({"status": "paused"})


@mcp.tool()
def next_track() -> str:
    """Skip to the next track in Apple Music."""
    ok, out = run_applescript('tell application "Music" to next track')
    if not ok:
        return error_response(out)
    return success_response({"status": "skipped_to_next"})


@mcp.tool()
def previous_track() -> str:
    """Go to the previous track in Apple Music."""
    ok, out = run_applescript('tell application "Music" to previous track')
    if not ok:
        return error_response(out)
    return success_response({"status": "skipped_to_previous"})


@mcp.tool()
def set_volume(volume: int) -> str:
    """Set Apple Music volume (0-100)."""
    if not 0 <= volume <= 100:
        return error_response("Volume must be between 0 and 100")
    ok, out = run_applescript(f'tell application "Music" to set sound volume to {volume}')
    if not ok:
        return error_response(out)
    return success_response({"volume": volume})


@mcp.tool()
def set_shuffle(enabled: bool) -> str:
    """Toggle shuffle mode in Apple Music."""
    val = "true" if enabled else "false"
    ok, out = run_applescript(f'tell application "Music" to set shuffle enabled to {val}')
    if not ok:
        return error_response(out)
    return success_response({"shuffle": enabled})


@mcp.tool()
def set_repeat(mode: str) -> str:
    """Set repeat mode in Apple Music. Mode must be 'off', 'one', or 'all'."""
    mode_map = {"off": "off", "one": "one", "all": "all"}
    if mode not in mode_map:
        return error_response("Mode must be 'off', 'one', or 'all'")
    ok, out = run_applescript(f'tell application "Music" to set song repeat to {mode}')
    if not ok:
        return error_response(out)
    return success_response({"repeat": mode})


# ---------------------------------------------------------------------------
# Info Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_now_playing() -> str:
    """Get the currently playing track info from Apple Music."""
    script = '''
tell application "Music"
    if player state is stopped then
        return "STATE:stopped"
    end if
    set trackName to name of current track
    set trackArtist to artist of current track
    set trackAlbum to album of current track
    set trackDuration to duration of current track
    set playerPos to player position
    set playerSt to player state as string
    set vol to sound volume
    return trackName & "|||" & trackArtist & "|||" & trackAlbum & "|||" & trackDuration & "|||" & playerPos & "|||" & playerSt & "|||" & vol
end tell
'''
    ok, out = run_applescript(script)
    if not ok:
        return error_response(out)
    if out == "STATE:stopped":
        return success_response({"state": "stopped"})
    parts = out.split("|||")
    if len(parts) < 7:
        return error_response(f"Unexpected output: {out}")
    return success_response({
        "name": parts[0],
        "artist": parts[1],
        "album": parts[2],
        "duration": float(parts[3]),
        "position": float(parts[4]),
        "state": parts[5],
        "volume": int(parts[6])
    })


@mcp.tool()
def get_volume() -> str:
    """Get the current Apple Music volume level."""
    ok, out = run_applescript('tell application "Music" to get sound volume')
    if not ok:
        return error_response(out)
    return success_response({"volume": int(out)})


# ---------------------------------------------------------------------------
# Search Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_library(query: str, limit: int = 10) -> str:
    """Search the user's local Apple Music library. Use this to get database IDs needed for playback. For multiple songs, pass the database IDs to create_playlist. For a single song, use play_song."""
    script = f'''
tell application "Music"
    set results to (search playlist "Library" for "{query}" only songs)
    set output to ""
    set maxCount to {limit}
    set i to 0
    repeat with t in results
        if i >= maxCount then exit repeat
        set trackInfo to (database ID of t) & "|||" & (name of t) & "|||" & (artist of t) & "|||" & (album of t) & "|||" & (duration of t)
        if i > 0 then set output to output & "\\n"
        set output to output & trackInfo
        set i to i + 1
    end repeat
    return output
end tell
'''
    ok, out = run_applescript(script)
    if not ok:
        return error_response(out)
    if not out:
        return success_response({"tracks": []})
    tracks = []
    for line in out.split("\n"):
        parts = line.split("|||")
        if len(parts) >= 5:
            tracks.append({
                "database_id": int(parts[0]),
                "name": parts[1],
                "artist": parts[2],
                "album": parts[3],
                "duration": float(parts[4])
            })
    return success_response({"tracks": tracks})


@mcp.tool()
def search_catalog(query: str, limit: int = 10) -> str:
    """Search the full Apple Music catalog via iTunes Search API for song discovery. Use this to find song names/artists, then use search_library to find matching tracks in the user's library for playback."""
    encoded = urllib.parse.quote(query)
    url = f"https://itunes.apple.com/search?term={encoded}&media=music&entity=song&limit={limit}"
    try:
        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(url, headers={"User-Agent": "MCP-AppleMusic/1.0"})
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return error_response(f"Search failed: {e}")
    tracks = []
    for r in data.get("results", []):
        tracks.append({
            "name": r.get("trackName"),
            "artist": r.get("artistName"),
            "album": r.get("collectionName"),
            "duration_ms": r.get("trackTimeMillis"),
            "apple_music_url": r.get("trackViewUrl"),
            "preview_url": r.get("previewUrl")
        })
    return success_response({"tracks": tracks})


# ---------------------------------------------------------------------------
# Queue / Playlist Tools
# ---------------------------------------------------------------------------

QUEUE_PLAYLIST = "Claude DJ Queue"


def ensure_queue_playlist() -> tuple[bool, str]:
    """Create the queue playlist if it doesn't exist."""
    script = f'''
tell application "Music"
    if not (exists playlist "{QUEUE_PLAYLIST}") then
        make new playlist with properties {{name:"{QUEUE_PLAYLIST}"}}
    end if
    return "ok"
end tell
'''
    return run_applescript(script)


@mcp.tool()
def play_song(database_id: int = 0) -> str:
    """Play a specific song by its database ID from the library. Use search_library first to get the database ID."""
    if not database_id:
        return error_response("Provide a database_id. Use search_library to find tracks first.")
    script = f'''
tell application "Music"
    set theTracks to (every track of playlist "Library" whose database ID is {database_id})
    if (count of theTracks) > 0 then
        play item 1 of theTracks
        return "playing"
    else
        return "not_found"
    end if
end tell
'''
    ok, out = run_applescript(script)
    if not ok:
        return error_response(out)
    if out == "not_found":
        return error_response("Track not found in library")
    return success_response({"status": "playing", "database_id": database_id})


@mcp.tool()
def play_next(database_id: int = 0) -> str:
    """Immediately play a specific song, interrupting current playback. Use search_library first to get the database ID."""
    if not database_id:
        return error_response("Provide a database_id. Use search_library to find tracks first.")
    script = f'''
tell application "Music"
    set theTracks to (every track of playlist "Library" whose database ID is {database_id})
    if (count of theTracks) > 0 then
        play item 1 of theTracks
        return "playing"
    else
        return "not_found"
    end if
end tell
'''
    ok, out = run_applescript(script)
    if not ok:
        return error_response(out)
    if out == "not_found":
        return error_response("Track not found in library")
    return success_response({"status": "now_playing", "database_id": database_id})


@mcp.tool()
def queue_song(database_id: int = 0) -> str:
    """Add a song to the 'Claude DJ Queue' playlist by database ID. Use search_library first to get the database ID."""
    if not database_id:
        return error_response("Provide a database_id. Use search_library to find tracks first.")
    ok, msg = ensure_queue_playlist()
    if not ok:
        return error_response(f"Failed to create queue playlist: {msg}")

    script = f'''
tell application "Music"
    set theTracks to (every track of playlist "Library" whose database ID is {database_id})
    if (count of theTracks) > 0 then
        duplicate item 1 of theTracks to playlist "{QUEUE_PLAYLIST}"
        return "queued"
    else
        return "not_found"
    end if
end tell
'''
    ok, out = run_applescript(script)
    if not ok:
        return error_response(out)
    if out == "not_found":
        return error_response("Track not found in library")
    return success_response({"status": "queued", "database_id": database_id})


@mcp.tool()
def clear_queue() -> str:
    """Clear all tracks from the 'Claude DJ Queue' playlist. Use this before switching genres or building a new queue."""
    script = f'''
tell application "Music"
    if (exists playlist "{QUEUE_PLAYLIST}") then
        delete playlist "{QUEUE_PLAYLIST}"
        return "cleared"
    else
        return "no_queue"
    end if
end tell
'''
    ok, out = run_applescript(script)
    if not ok:
        return error_response(out)
    if out == "no_queue":
        return success_response({"status": "queue_already_empty"})
    return success_response({"status": "queue_cleared"})


@mcp.tool()
def get_queue() -> str:
    """List all tracks in the 'Claude DJ Queue' playlist."""
    ok, msg = ensure_queue_playlist()
    if not ok:
        return error_response(f"Failed to access queue playlist: {msg}")

    script = f'''
tell application "Music"
    if not (exists playlist "{QUEUE_PLAYLIST}") then
        return ""
    end if
    set output to ""
    set i to 0
    repeat with t in tracks of playlist "{QUEUE_PLAYLIST}"
        set trackInfo to (database ID of t) & "|||" & (name of t) & "|||" & (artist of t) & "|||" & (album of t)
        if i > 0 then set output to output & "\\n"
        set output to output & trackInfo
        set i to i + 1
    end repeat
    return output
end tell
'''
    ok, out = run_applescript(script)
    if not ok:
        return error_response(out)
    if not out:
        return success_response({"tracks": []})
    tracks = []
    for line in out.split("\n"):
        parts = line.split("|||")
        if len(parts) >= 4:
            tracks.append({
                "database_id": int(parts[0]),
                "name": parts[1],
                "artist": parts[2],
                "album": parts[3]
            })
    return success_response({"tracks": tracks})


@mcp.tool()
def create_playlist(name: str, track_ids: list[int]) -> str:
    """Primary method for playing multiple songs. Creates a playlist from library track database IDs and immediately starts playing it. Use search_library to get database IDs first."""
    if not track_ids:
        return error_response("No track IDs provided")
    # Build AppleScript to create playlist and add tracks
    add_lines = ""
    for tid in track_ids:
        add_lines += f'''
        set matchedTracks to (every track of playlist "Library" whose database ID is {tid})
        if (count of matchedTracks) > 0 then
            duplicate item 1 of matchedTracks to newPlaylist
        end if
'''
    script = f'''
tell application "Music"
    if (exists playlist "{name}") then
        delete playlist "{name}"
    end if
    set newPlaylist to (make new playlist with properties {{name:"{name}"}})
    {add_lines}
    play newPlaylist
    return "created_and_playing"
end tell
'''
    ok, out = run_applescript(script)
    if not ok:
        return error_response(out)
    return success_response({"status": "created_and_playing", "name": name, "track_count": len(track_ids)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")

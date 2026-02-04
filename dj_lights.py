#!/usr/bin/env python3
"""
DJ Lights - Real-time music visualization for Kasa smart bulbs.
Captures audio, analyzes frequencies via FFT, and maps them to colors.

Environment Variables:
    BULBS - Comma or colon separated list of bulb IPs
           Format: "name1:ip1,name2:ip2" or just "ip1,ip2"
           Example: "left:192.168.0.103,right:192.168.0.104" """

import argparse
import asyncio
import colorsys
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

# Try to use gnureadline for better macOS support, fall back to readline
try:
    import gnureadline as readline
except ImportError:
    import readline

import argcomplete
import mss
import numpy as np
import Quartz
import sounddevice as sd
from dotenv import load_dotenv
from kasa import Discover
from kasa.module import Module
from PIL import Image
from scipy.fft import rfft, rfftfreq
from sklearn.cluster import KMeans

load_dotenv()
# Audio settings
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048  # ~46ms of audio
UPDATE_RATE = 20  # Updates per second to bulbs

# Frequency bands (Hz)
BASS_RANGE = (20, 250)
MID_RANGE = (250, 2000)
TREBLE_RANGE = (2000, 8000)


def parse_bulbs_env():
    """Parse BULBS environment variable into a dict of name:ip pairs."""
    env_val = os.environ.get("BULBS", "")
    if not env_val:
        print("Warning: BULBS environment variable not set!")
        print("Set it like: export BULBS='left:192.168.0.103,right:192.168.0.104'")
        print("Or just IPs: export BULBS='192.168.0.103,192.168.0.104'")
        return {}

    bulbs = {}
    # Split by comma
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    for i, part in enumerate(parts):
        if ":" in part:
            # Format: name:ip
            name, ip = part.split(":", 1)
            bulbs[name.strip()] = ip.strip()
        else:
            # Just IP, auto-name
            bulbs[f"bulb{i+1}"] = part.strip()
    return bulbs


def run_applescript(script):
    """Run AppleScript and return output."""
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"AppleScript error: {e}", file=sys.stderr)
        return None


def get_current_track_info():
    """Get current playing track info from Apple Music."""
    script = '''
    tell application "Music"
        if player state is playing then
            return name of current track & "|" & artist of current track
        end if
    end tell
    '''
    result = run_applescript(script)
    if result:
        parts = result.split("|")
        if len(parts) == 2:
            return {"track": parts[0], "artist": parts[1]}
    return None


def get_album_artwork():
    """Extract current track's album artwork to a temp file."""
    temp_file = tempfile.mktemp(suffix='.jpg')

    script = f'''
    tell application "Music"
        if player state is playing then
            try
                set artworkData to data of artwork 1 of current track
                set fileRef to open for access POSIX file "{temp_file}" with write permission
                write artworkData to fileRef
                close access fileRef
                return "{temp_file}"
            end try
        end if
    end tell
    '''

    result = run_applescript(script)
    if result and Path(temp_file).exists():
        return temp_file
    return None


def generate_analogous_colors(base_hue, base_sat, count=4):
    """Generate analogous colors (similar shades) based on a base hue."""
    colors = [(base_hue, base_sat)]

    # Analogous colors are close on the color wheel (30-45° apart)
    offsets = [30, -30, 60, -60, 45, -45]

    for i, offset in enumerate(offsets):
        if len(colors) >= count:
            break
        colors.append(((base_hue + offset) % 360, base_sat))

    return colors[:count]


def extract_dominant_colors(image_path, num_colors=5):
    """Extract dominant colors from image, return as (hue, saturation) tuples."""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((150, 150))  # Resize for faster processing

        # Get pixel data
        pixels = np.array(img).reshape(-1, 3)

        # Filter out very dark pixels (they don't make good light colors)
        brightness = pixels.mean(axis=1)
        pixels = pixels[brightness > 30]

        if len(pixels) < num_colors:
            # Not enough pixels, generate colors
            return generate_analogous_colors(200, 70, 4)  # Default blue-ish

        # K-means clustering to find dominant colors
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_

        # Convert RGB to HSV and check if image is grayscale
        hsv_colors = []
        total_saturation = 0
        for r, g, b in colors:
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hsv_colors.append((h, s, v))
            total_saturation += s

        avg_saturation = total_saturation / len(hsv_colors)

        # If average saturation is very low, image is grayscale
        # Generate a nice palette instead of using meaningless hue values
        if avg_saturation < 0.15:
            # Grayscale album - use a cool blue/purple palette
            return generate_analogous_colors(220, 70, 4)

        # Extract colors with meaningful saturation
        color_data = []
        for h, s, v in hsv_colors:
            if v > 0.2 and s > 0.1:  # Filter by brightness and minimum saturation
                hue = int(h * 360)
                sat = max(60, int(s * 100))  # Boost saturation for visibility
                color_data.append((hue, sat))

        # If we didn't get enough colorful colors, generate from what we have
        if len(color_data) < 3:
            if len(color_data) >= 1:
                # Use the first extracted color as base
                base_hue, base_sat = color_data[0]
                color_data = generate_analogous_colors(base_hue, max(70, base_sat), 4)
            else:
                # No colorful colors found, use a nice default
                color_data = generate_analogous_colors(220, 70, 4)

        return sorted(color_data, key=lambda x: x[0])  # Sort by hue
    except Exception as e:
        print(f"Error extracting colors: {e}", file=sys.stderr)
        return generate_analogous_colors(220, 70, 4)  # Fallback


def list_app_windows(app_name="Google Chrome"):
    """List all windows for an application.

    Returns list of dicts with window info: {id, title, width, height, bounds}
    """
    try:
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly,
            Quartz.kCGNullWindowID
        )
        app_windows = []
        for win in windows:
            owner = win.get('kCGWindowOwnerName', '')
            if app_name.lower() in owner.lower():
                bounds = win.get('kCGWindowBounds')
                if bounds:
                    width = int(bounds['Width'])
                    height = int(bounds['Height'])
                    # Skip tiny windows (like menu bar items)
                    if width > 200 and height > 200:
                        title = win.get('kCGWindowName', '') or '(untitled)'
                        window_id = win.get('kCGWindowNumber', 0)
                        app_windows.append({
                            'id': window_id,
                            'title': title[:50] + '...' if len(title) > 50 else title,
                            'width': width,
                            'height': height,
                            'bounds': {
                                'left': int(bounds['X']),
                                'top': int(bounds['Y']),
                                'width': width,
                                'height': height
                            }
                        })
        return app_windows
    except Exception as e:
        print(f"Error listing windows: {e}", file=sys.stderr)
        return []


async def pick_window_interactive(app_name="Google Chrome"):
    """Interactively prompt user to select a window.

    Returns the window title for matching, or None if cancelled.
    """
    windows = list_app_windows(app_name)
    if not windows:
        print(f"\nNo windows found for '{app_name}'")
        print("Make sure the app is open and has visible windows.")
        return None

    # Sort by size (largest first)
    windows.sort(key=lambda w: w['width'] * w['height'], reverse=True)

    print(f"\nSelect a {app_name} window to capture:")
    print("-" * 60)
    for i, win in enumerate(windows):
        print(f"  [{i}] {win['title']}")
        print(f"      Size: {win['width']}x{win['height']}")
    print("-" * 60)

    # Get user input
    try:
        choice = await asyncio.get_event_loop().run_in_executor(
            None, input, "Enter number (or 'q' to cancel): "
        )
        choice = choice.strip().lower()

        if choice == 'q' or choice == '':
            print("Cancelled.")
            return None

        idx = int(choice)
        if 0 <= idx < len(windows):
            selected = windows[idx]
            print(f"\nSelected: {selected['title']}")
            return selected['title']
        else:
            print(f"Invalid selection. Please enter 0-{len(windows)-1}")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None
    except EOFError:
        return None


def find_window(app_name="Google Chrome", window_title=None, window_index=0):
    """Find window bounds for a specific application.

    Uses macOS Quartz APIs to list windows and find one matching the app name.
    Args:
        app_name: Application name to match
        window_title: If provided, find window with this title (partial match, case-insensitive)
        window_index: Fallback - which window to use if multiple exist (0 = first/largest)
    Returns dict with {left, top, width, height, id, title} or None if not found.
    """
    windows = list_app_windows(app_name)
    if not windows:
        return None

    # Sort by size (largest first) for consistent ordering
    windows.sort(key=lambda w: w['width'] * w['height'], reverse=True)

    # If window_title provided, find by title (partial match, case-insensitive)
    if window_title:
        title_lower = window_title.lower()
        for win in windows:
            if title_lower in win['title'].lower():
                return {
                    'left': win['bounds']['left'],
                    'top': win['bounds']['top'],
                    'width': win['width'],
                    'height': win['height'],
                    'id': win['id'],
                    'title': win['title']
                }
        # Title not found, fall through to index-based selection
        return None

    # Fallback to index-based selection
    if window_index >= len(windows):
        window_index = 0

    win = windows[window_index]
    return {
        'left': win['bounds']['left'],
        'top': win['bounds']['top'],
        'width': win['width'],
        'height': win['height'],
        'id': win['id'],
        'title': win['title']
    }


class AudioAnalyzer:
    """Captures and analyzes audio in real-time."""

    def __init__(self, device=None):
        self.device = device
        self.buffer = deque(maxlen=5)
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        if len(indata.shape) > 1:
            audio = np.mean(indata, axis=1)
        else:
            audio = indata.flatten()
        self.buffer.append(audio)

    def start(self):
        self.stream = sd.InputStream(
            device=self.device,
            channels=2,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            callback=self.audio_callback,
        )
        self.stream.start()
        print(f"Audio stream started on device: {self.device or 'default'}")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_frequency_bands(self):
        if not self.buffer:
            return {"bass": 0, "mid": 0, "treble": 0, "energy": 0}

        audio = np.concatenate(list(self.buffer))
        window = np.hanning(len(audio))
        audio = audio * window

        fft_data = np.abs(rfft(audio))
        freqs = rfftfreq(len(audio), 1 / SAMPLE_RATE)

        def band_energy(low, high):
            mask = (freqs >= low) & (freqs <= high)
            return np.mean(fft_data[mask]) if np.any(mask) else 0

        return {
            "bass": band_energy(*BASS_RANGE),
            "mid": band_energy(*MID_RANGE),
            "treble": band_energy(*TREBLE_RANGE),
            "energy": np.mean(fft_data),
        }


class ColorMapper:
    """Base color mapper with auto-gain normalization."""

    def __init__(self, sensitivity=1.0, max_brightness=100):
        self.peak_bass = 0.1
        self.peak_mid = 0.1
        self.peak_treble = 0.1
        self.peak_energy = 0.1
        self.decay = 0.995
        self.last_bass = 0
        self.beat_count = 0
        # Sensitivity: 0.5 = less sensitive (fewer beats), 2.0 = more sensitive (more beats)
        self.sensitivity = max(0.1, min(3.0, sensitivity))
        # Max brightness cap (1-100)
        self.max_brightness = max(1, min(100, max_brightness))

    def cap_brightness(self, brightness):
        """Apply max brightness cap."""
        return min(brightness, self.max_brightness)

    def normalize(self, bands):
        bass, mid, treble, energy = (
            bands["bass"],
            bands["mid"],
            bands["treble"],
            bands["energy"],
        )
        self.peak_bass = max(self.peak_bass * self.decay, bass, 0.01)
        self.peak_mid = max(self.peak_mid * self.decay, mid, 0.01)
        self.peak_treble = max(self.peak_treble * self.decay, treble, 0.01)
        self.peak_energy = max(self.peak_energy * self.decay, energy, 0.001)
        return {
            "bass": min(bass / self.peak_bass, 1.0),
            "mid": min(mid / self.peak_mid, 1.0),
            "treble": min(treble / self.peak_treble, 1.0),
            "energy": min(energy / self.peak_energy, 1.0),
        }

    def detect_beat(self, norm_bass):
        bass_delta = norm_bass - self.last_bass
        # Adjust thresholds based on sensitivity
        # Lower threshold = more beats detected
        delta_threshold = 0.3 / self.sensitivity  # 0.3 default, lower with higher sensitivity
        level_threshold = 0.5 / self.sensitivity  # 0.5 default
        is_beat = bass_delta > delta_threshold and norm_bass > level_threshold
        self.last_bass = norm_bass * 0.7 + self.last_bass * 0.3
        if is_beat:
            self.beat_count += 1
        return is_beat


class ReactiveMapper(ColorMapper):
    """Default reactive mode - full spectrum, beat-driven hue shifts."""

    def __init__(self, sensitivity=1.0, max_brightness=100):
        super().__init__(sensitivity, max_brightness)
        self.hue_base = 0

    def map_to_hsv(self, bands):
        n = self.normalize(bands)
        is_beat = self.detect_beat(n["bass"])

        if is_beat:
            self.hue_base = (self.hue_base + 60 + int(n["treble"] * 60)) % 360

        freq_balance = n["treble"] - n["bass"]
        hue = (self.hue_base + int(freq_balance * 90)) % 360
        saturation = 70 + int(n["energy"] * 30)
        brightness = 40 + int(n["bass"] * 40) + int(n["energy"] * 20)
        if is_beat:
            brightness = min(100, brightness + 20)

        return hue, min(100, saturation), self.cap_brightness(min(100, brightness)), is_beat


class AlbumReactiveMapper(ColorMapper):
    """Album-reactive mode - adapts to song vibe (chill vs hype)."""

    def __init__(self, color_palette=None, sensitivity=1.0, max_brightness=100):
        super().__init__(sensitivity, max_brightness)
        # Color palette stores (hue, sat) tuples
        self.color_palette = color_palette or [(0, 70), (120, 70), (240, 70)]
        self.current_color_index = 0
        self.hue_base, self.sat_base = self.color_palette[0]
        self.last_track_check = 0
        self.current_track = None

        # Smooth transitions
        self.current_hue = self.hue_base
        self.current_sat = self.sat_base
        self.current_bright = 60

        # Song vibe detection (rolling average of energy)
        self.energy_history = deque(maxlen=100)  # Last ~5 seconds of energy
        self.song_vibe = 0.5  # 0 = super chill, 1 = hyper

        # Beatdrop detection
        self.recent_beats = deque(maxlen=10)
        self.is_beatdrop = False
        self.beatdrop_intensity = 0

    def update_palette(self, new_palette):
        """Update color palette (called when track changes)."""
        if new_palette and len(new_palette) > 0:
            self.color_palette = new_palette
            # Reset index to 0 when palette changes (avoids IndexError)
            self.current_color_index = 0
            self.hue_base, self.sat_base = self.color_palette[0]
            # Reset vibe when track changes
            self.energy_history.clear()
            self.song_vibe = 0.5

    def calculate_song_vibe(self, energy):
        """Determine if song is chill (0) or hype (1) based on energy history."""
        self.energy_history.append(energy)
        if len(self.energy_history) < 20:
            return 0.5  # Not enough data, assume medium

        # Calculate average energy
        avg_energy = sum(self.energy_history) / len(self.energy_history)

        # Also consider variance (chill songs = low variance, hype = high variance)
        variance = sum((e - avg_energy) ** 2 for e in self.energy_history) / len(self.energy_history)

        # Vibe = combination of energy level and variance
        # High energy + high variance = hype (closer to 1)
        # Low energy + low variance = chill (closer to 0)
        vibe = (avg_energy * 0.7) + (variance * 0.3)
        return max(0, min(1, vibe))

    def detect_beatdrop(self, vibe):
        """Detect beatdrop - threshold adapts to song vibe."""
        if len(self.recent_beats) < 5:
            return False

        time_span = self.recent_beats[-1] - self.recent_beats[0]

        # Adaptive threshold: chill songs need faster beats to trigger
        # Chill (vibe=0.2): need beats in < 1.0s (super tight)
        # Hype (vibe=0.8): beats in < 2.5s triggers (easier)
        threshold = 1.0 + (vibe * 1.5)  # 1.0-2.5 seconds

        if time_span < threshold and len(self.recent_beats) >= 5:
            self.beatdrop_intensity = min(100, int((5 / time_span) * 20))
            return True
        return False

    def map_to_hsv(self, bands):
        n = self.normalize(bands)
        is_beat = self.detect_beat(n["bass"])
        current_time = time.time()

        # Update song vibe
        self.song_vibe = self.calculate_song_vibe(n["energy"])

        # Beat handling - only change color on beat if energy is high enough
        if is_beat:
            self.recent_beats.append(current_time)
            # Chill songs: stay on same color longer
            # Hype songs: change color on every beat
            if n["energy"] > (0.4 - self.song_vibe * 0.2):
                self.current_color_index = (self.current_color_index + 1) % len(self.color_palette)
                self.hue_base, self.sat_base = self.color_palette[self.current_color_index]

        # Detect beatdrop with adaptive threshold
        self.is_beatdrop = self.detect_beatdrop(self.song_vibe)

        # === HUE === Smooth transition to target
        freq_balance = n["treble"] - n["bass"]

        # Scale hue variation based on vibe
        # Chill (0.2): ±5° variation
        # Hype (0.8): ±30° variation
        hue_range = 5 + (self.song_vibe * 25)
        if self.is_beatdrop:
            hue_range *= 1.5  # Extra variation during drops

        target_hue = (self.hue_base + int(freq_balance * hue_range)) % 360

        # Smooth lerp to target (chill = slower, hype = faster)
        lerp_speed = 0.05 + (self.song_vibe * 0.15)  # 0.05-0.20
        self.current_hue += ((target_hue - self.current_hue) * lerp_speed)
        self.current_hue = self.current_hue % 360

        # === SATURATION === Use album color, subtle variation
        sat_variation = int(n["energy"] * 10 * self.song_vibe)  # 0-10 based on vibe
        target_sat = self.sat_base + sat_variation

        if self.is_beatdrop and self.song_vibe > 0.6:
            # Only boost sat on beatdrops for hype songs
            target_sat = min(100, target_sat + 20)

        # Smooth lerp
        self.current_sat += ((target_sat - self.current_sat) * 0.08)

        # === BRIGHTNESS === Adaptive to vibe
        base_bright = 45 + int(n["energy"] * 20)
        bass_influence = int(n["bass"] * 20 * self.song_vibe)  # Scale bass impact by vibe

        target_bright = base_bright + bass_influence

        # Beat brightness spike - scaled by vibe
        if is_beat:
            spike = 15 + int(self.song_vibe * 25)  # 15-40 spike
            target_bright = min(100, target_bright + spike)

        if self.is_beatdrop and self.song_vibe > 0.6:
            target_bright = min(100, target_bright + 15)

        # Smooth lerp (chill = slower, hype = faster)
        bright_lerp = 0.08 + (self.song_vibe * 0.12)  # 0.08-0.20
        self.current_bright += ((target_bright - self.current_bright) * bright_lerp)

        return int(self.current_hue), int(self.current_sat), self.cap_brightness(int(self.current_bright)), is_beat


class CalmMapper(ColorMapper):
    """Calm mode - slow, gradual, cool chill colors (blues, greens, cyans)."""

    def __init__(self, sensitivity=1.0, max_brightness=100):
        super().__init__(sensitivity, max_brightness)
        # Cool color palette: greens (120), cyans (180), blues (210-240)
        self.hue = 180  # Start cyan
        self.target_hue = 180
        self.saturation = 50
        self.target_sat = 50
        self.brightness = 60
        self.target_bright = 60

    def map_to_hsv(self, bands):
        n = self.normalize(bands)
        is_beat = self.detect_beat(n["bass"])

        # Gentle hue movement within cool spectrum (90-250: green to blue)
        # Bass pulls toward deeper blues, treble toward greens/cyans
        base_hue = 170  # Center around cyan
        freq_influence = (n["treble"] - n["bass"]) * 60  # ±60 degrees
        self.target_hue = base_hue + freq_influence + (n["mid"] * 30)

        # Clamp to cool spectrum (90-250)
        self.target_hue = max(90, min(250, self.target_hue))

        # On beats, gently shift toward a complementary cool color
        if is_beat:
            self.target_hue = 90 + ((self.target_hue - 90 + 80) % 160)

        # Very smooth transitions (slow lerp)
        self.hue += (self.target_hue - self.hue) * 0.03

        # Saturation: lower when calm, slightly higher with energy
        self.target_sat = 35 + int(n["energy"] * 35)  # 35-70%
        self.saturation += (self.target_sat - self.saturation) * 0.05

        # Brightness: gentle pulsing with music
        self.target_bright = 45 + int(n["energy"] * 25) + int(n["bass"] * 15)
        self.brightness += (self.target_bright - self.brightness) * 0.04

        return int(self.hue) % 360, int(self.saturation), self.cap_brightness(int(self.brightness)), False


class RaveMapper(ColorMapper):
    """Rave mode - aggressive, fast, full spectrum, strobe on beats."""

    def __init__(self, sensitivity=1.0, max_brightness=100):
        super().__init__(sensitivity, max_brightness)
        self.hue = 0
        self.strobe_frames = 0

    def map_to_hsv(self, bands):
        n = self.normalize(bands)
        is_beat = self.detect_beat(n["bass"])

        # Rapid hue cycling
        self.hue = (self.hue + 5 + int(n["energy"] * 15)) % 360

        if is_beat:
            # Jump to contrasting color
            self.hue = (self.hue + 120) % 360
            self.strobe_frames = 3

        # Strobe effect on beats
        if self.strobe_frames > 0:
            self.strobe_frames -= 1
            brightness = 100 if self.strobe_frames % 2 == 0 else 50
        else:
            brightness = 60 + int(n["bass"] * 40)

        saturation = 100  # Max saturation always

        return self.hue, saturation, self.cap_brightness(min(100, brightness)), is_beat


class FixedHueMapper(ColorMapper):
    """Fixed hue mode - constant hue, varies only saturation and brightness."""

    def __init__(self, hue=0, sensitivity=1.0, max_brightness=100):
        super().__init__(sensitivity, max_brightness)
        self.fixed_hue = hue % 360
        self.saturation = 70
        self.brightness = 60
        self.target_sat = 70
        self.target_bright = 60

    def map_to_hsv(self, bands):
        n = self.normalize(bands)
        is_beat = self.detect_beat(n["bass"])

        # Saturation: responsive to overall energy
        # Higher energy = higher saturation
        self.target_sat = 40 + int(n["energy"] * 60)  # 40-100%

        # Brightness: responsive to bass and energy
        base_bright = 40 + int(n["energy"] * 30)  # 40-70
        bass_influence = int(n["bass"] * 30)  # 0-30

        self.target_bright = base_bright + bass_influence

        # Beat brightness spike
        if is_beat:
            self.target_bright = min(100, self.target_bright + 25)

        # Smooth transitions
        self.saturation += (self.target_sat - self.saturation) * 0.12
        self.brightness += (self.target_bright - self.brightness) * 0.15

        return self.fixed_hue, int(self.saturation), self.cap_brightness(int(self.brightness)), is_beat


class MovieMapper(ColorMapper):
    """Movie mode - captures window colors and maps to dual bulbs (left/right)."""

    def __init__(self, sensitivity=1.0, audio_blend=0.3, window_name="Google Chrome", window_title=None, max_brightness=100):
        super().__init__(sensitivity, max_brightness)
        self.sct = mss.mss()
        self.window_name = window_name
        self.target_window_title = window_title  # Title to search for (partial match)
        self.audio_blend = audio_blend  # 0 = pure screen, 1 = audio affects brightness

        # Left bulb colors
        self.left_hue = 0
        self.left_sat = 50
        self.left_bright = 60
        self.target_left_hue = 0
        self.target_left_sat = 50
        self.target_left_bright = 60

        # Right bulb colors
        self.right_hue = 180
        self.right_sat = 50
        self.right_bright = 60
        self.target_right_hue = 180
        self.target_right_sat = 50
        self.target_right_bright = 60

        # Capture timing
        self.last_capture = 0
        self.capture_interval = 0.1  # ~10 FPS
        self.window_bounds = None
        self.current_window_title = None  # Title of the currently captured window

    def extract_region_color(self, img_array, x_start, x_end, y_start, y_end):
        """Extract average color from a region of the image."""
        region = img_array[y_start:y_end, x_start:x_end]
        if region.size == 0:
            return 0, 50, 60  # Default HSV

        # Average RGB
        avg_r = np.mean(region[:, :, 2])  # mss uses BGRA
        avg_g = np.mean(region[:, :, 1])
        avg_b = np.mean(region[:, :, 0])

        # Convert to HSV
        h, s, v = colorsys.rgb_to_hsv(avg_r / 255, avg_g / 255, avg_b / 255)
        return int(h * 360), int(s * 100), int(v * 100)

    def capture_window_colors(self):
        """Capture the target window and extract left/right region colors."""
        current_time = time.time()

        # Only capture at specified interval
        if current_time - self.last_capture < self.capture_interval:
            return None, None

        self.last_capture = current_time

        # Find window bounds (re-check in case window moved/resized)
        # Use title matching if a target title is set, otherwise fall back to largest window
        bounds = find_window(self.window_name, window_title=self.target_window_title)
        if not bounds:
            return None, None

        self.window_bounds = bounds
        self.current_window_title = bounds.get('title', '')

        try:
            # Capture the window
            screenshot = self.sct.grab(bounds)
            img_array = np.array(screenshot)

            height = bounds['height']
            width = bounds['width']

            # Left region: left 1/2 of window
            left_x_end = width // 2
            left_color = self.extract_region_color(
                img_array, 0, left_x_end, 0, height
            )

            # Right region: right 1/2 of window
            right_x_start = width // 2
            right_color = self.extract_region_color(
                img_array, right_x_start, width, 0, height
            )

            return left_color, right_color

        except Exception as e:
            print(f"\rCapture error: {e}", end="", file=sys.stderr)
            return None, None

    def map_to_dual_hsv(self, bands=None):
        """Map audio and screen colors to dual HSV values (left, right).

        Args:
            bands: Audio frequency bands dict. If None, only screen colors are used.
        """
        is_beat = False

        # Process audio if available
        if bands is not None and self.audio_blend > 0:
            n = self.normalize(bands)
            is_beat = self.detect_beat(n["bass"])
        else:
            n = None

        # Capture screen colors
        left_color, right_color = self.capture_window_colors()

        # Update targets if we got new colors
        if left_color:
            self.target_left_hue, self.target_left_sat, self.target_left_bright = left_color
        if right_color:
            self.target_right_hue, self.target_right_sat, self.target_right_bright = right_color

        # Smooth lerp to targets
        lerp_speed = 0.15
        self.left_hue += (self.target_left_hue - self.left_hue) * lerp_speed
        self.left_sat += (self.target_left_sat - self.left_sat) * lerp_speed
        self.left_bright += (self.target_left_bright - self.left_bright) * lerp_speed

        self.right_hue += (self.target_right_hue - self.right_hue) * lerp_speed
        self.right_sat += (self.target_right_sat - self.right_sat) * lerp_speed
        self.right_bright += (self.target_right_bright - self.right_bright) * lerp_speed

        # Audio influence on brightness (if audio_blend > 0 and we have audio)
        if n is not None and self.audio_blend > 0:
            # Bass boosts brightness
            bass_boost = int(n["bass"] * 20 * self.audio_blend)
            energy_boost = int(n["energy"] * 10 * self.audio_blend)

            self.left_bright = min(100, self.left_bright + bass_boost + energy_boost)
            self.right_bright = min(100, self.right_bright + bass_boost + energy_boost)

            # Beat pulse
            if is_beat:
                beat_boost = int(20 * self.audio_blend)
                self.left_bright = min(100, self.left_bright + beat_boost)
                self.right_bright = min(100, self.right_bright + beat_boost)

        # Ensure minimum brightness for visibility
        self.left_bright = max(30, self.left_bright)
        self.right_bright = max(30, self.right_bright)

        left_hsv = (int(self.left_hue) % 360, int(self.left_sat), self.cap_brightness(int(self.left_bright)))
        right_hsv = (int(self.right_hue) % 360, int(self.right_sat), self.cap_brightness(int(self.right_bright)))

        return left_hsv, right_hsv, is_beat


class BulbController:
    """Controls Kasa smart bulbs."""

    def __init__(self, ips: dict):
        self.ips = ips
        self.bulbs = {}
        self.last_hsv = {}

    async def connect(self):
        print("Connecting to bulbs...")
        for name, ip in self.ips.items():
            try:
                bulb = await Discover.discover_single(ip)
                await bulb.update()
                self.bulbs[name] = bulb
                self.last_hsv[name] = (-999, -999, -999)  # Force first update
                print(f"Connected to {name}: {bulb.alias} at {ip}")
            except Exception as e:
                print(f"Failed to connect to {name} at {ip}: {e}")

        if not self.bulbs:
            raise RuntimeError("No bulbs connected!")

    async def disconnect(self):
        """Properly close all bulb connections."""
        for name, bulb in self.bulbs.items():
            await bulb.protocol.close()
        self.bulbs.clear()

    async def turn_on_all(self):
        for name, bulb in self.bulbs.items():
            await bulb.turn_on()
            print(f"Turned on {name}")

    async def turn_off_all(self):
        for name, bulb in self.bulbs.items():
            await bulb.turn_off()
            print(f"Turned off {name}")

    async def set_hsv(
        self, name: str, hue: int, saturation: int, brightness: int, debug=False
    ):
        if name not in self.bulbs:
            if debug:
                print(f"\nDebug: Bulb {name} not found in {list(self.bulbs.keys())}")
            return

        last = self.last_hsv[name]
        if (
            abs(hue - last[0]) < 5
            and abs(saturation - last[1]) < 5
            and abs(brightness - last[2]) < 5
        ):
            if debug:
                print(f"\nDebug: Skipping update for {name}, change too small")
            return

        try:
            light = self.bulbs[name].modules[Module.Light]
            await light.set_hsv(hue, saturation, brightness)
            self.last_hsv[name] = (hue, saturation, brightness)
            if debug:
                print(
                    f"\nDebug: Updated {name} to H:{hue} S:{saturation} B:{brightness}"
                )
        except Exception as e:
            print(f"\nError setting {name}: {e}", file=sys.stderr)
            if debug:
                import traceback

                traceback.print_exc()

    async def set_all_hsv(self, hue: int, saturation: int, brightness: int):
        tasks = [self.set_hsv(name, hue, saturation, brightness) for name in self.bulbs]
        await asyncio.gather(*tasks)

    async def set_complementary(self, hue: int, saturation: int, brightness: int):
        names = list(self.bulbs.keys())
        if len(names) >= 2:
            await asyncio.gather(
                self.set_hsv(names[0], hue, saturation, brightness),
                self.set_hsv(names[1], (hue + 180) % 360, saturation, brightness),
            )
        elif names:
            await self.set_hsv(names[0], hue, saturation, brightness)


def list_audio_devices():
    print("\nAvailable audio input devices:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " <-- (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{marker}")
    print("-" * 50)
    print("\nLook for 'BlackHole' or similar virtual audio device.")
    print("Use --device <number> to select.\n")


async def run_white_mode(controller, brightness: int, color_temp: str, disconnect=True):
    """Static warm white light mode."""
    # Color temperatures from cool to warm
    temps = {
        "cool2": (210, 25),  # Daylight blue
        "cool1": (180, 20),  # Cool white
        "neutral": (60, 15),  # Neutral white
        "warm1": (35, 30),  # Warm white
        "warm2": (25, 40),  # Candlelight
    }
    hue, sat = temps.get(color_temp, temps["neutral"])

    # Set the color_temp feature
    for name, bulb in controller.bulbs.items():
        light = bulb.modules[Module.Light]
        if light.has_feature("color_temp"):
            temp_feature = light.get_feature("color_temp")
            minimum_val = temp_feature.minimum_value
            maximum_val = temp_feature.maximum_value

            # Map color_temp to feature range
            color_range = maximum_val - minimum_val
            temp_map = {
                "cool2": minimum_val + color_range,
                "cool1": minimum_val + int(color_range * 0.75),
                "neutral": minimum_val + int(color_range * 0.5),
                "warm1": minimum_val + int(color_range * 0.25),
                "warm2": minimum_val,
            }
            temp_value = temp_map.get(color_temp, temp_map["neutral"])
            await light.set_color_temp(temp_value)
            await light.set_brightness(brightness)
            print(f"Set bulb {name} to color temperature '{temp_value}K'")
        else:
            print(f"Bulb {name} does not support color temperature feature.")
            # Fallback to HSV for this bulb
            await controller.set_hsv(name, hue, sat, brightness)

    if disconnect:
        await controller.disconnect()
    print("Done!")


async def run_visualization(
    controller, analyzer, mapper, color_mode: str, stop_event=None
):
    """Main visualization loop."""
    try:
        print("\nStarting audio capture...")
        print(f"Mode: {mapper.__class__.__name__}, Color mode: {color_mode}")
        if stop_event:
            print("Press Ctrl+C to stop.\n")
        else:
            print("Press Ctrl+C to stop.\n")

        analyzer.start()

        last_update = 0
        last_hue = 0
        last_track_check = 0
        current_track = None

        # Initial album artwork load for AlbumReactiveMapper
        if isinstance(mapper, AlbumReactiveMapper):
            print("Fetching album artwork...")
            artwork_path = get_album_artwork()
            if artwork_path:
                color_data = extract_dominant_colors(artwork_path)
                mapper.update_palette(color_data)
                track_info = get_current_track_info()
                if track_info:
                    current_track = track_info["track"]
                    print(f"Album colors from: {track_info['track']} - {track_info['artist']}")
                    # Format colors nicely
                    colors_str = ", ".join([f"H:{h}° S:{s}%" for h, s in color_data])
                    print(f"Color palette: {colors_str}\n")

        while True:
            if stop_event and stop_event.is_set():
                break

            # Check for track changes (every 5 seconds) for AlbumReactiveMapper
            now = time.time()
            if isinstance(mapper, AlbumReactiveMapper) and (now - last_track_check) > 5:
                last_track_check = now
                track_info = get_current_track_info()
                if track_info and track_info["track"] != current_track:
                    current_track = track_info["track"]
                    print(f"\n\nTrack changed: {track_info['track']} - {track_info['artist']}")
                    print("Updating album colors...")
                    artwork_path = get_album_artwork()
                    if artwork_path:
                        color_data = extract_dominant_colors(artwork_path)
                        mapper.update_palette(color_data)
                        colors_str = ", ".join([f"H:{h}° S:{s}%" for h, s in color_data])
                        print(f"New color palette: {colors_str}\n")

            bands = analyzer.get_frequency_bands()
            hue, sat, bright, is_beat = mapper.map_to_hsv(bands)

            beat_indicator = " BEAT!" if is_beat else ""
            beatdrop_indicator = ""
            vibe_indicator = ""

            if isinstance(mapper, AlbumReactiveMapper):
                if mapper.is_beatdrop:
                    beatdrop_indicator = " 🔥DROP"
                # Show vibe as emoji
                vibe = mapper.song_vibe
                if vibe < 0.3:
                    vibe_indicator = " 😌chill"
                elif vibe < 0.6:
                    vibe_indicator = " 🎵vibe "
                else:
                    vibe_indicator = " 🔊hype "

            print(
                f"\rHue: {hue:3d} | Sat: {sat:3d} | Bright: {bright:3d} | "
                f"Beats: {mapper.beat_count:4d}{beat_indicator:6s}{vibe_indicator}{beatdrop_indicator}",
                end="",
                flush=True,
            )

            now = time.time()
            time_since_update = now - last_update
            hue_change = abs(hue - last_hue)

            should_update = (
                time_since_update > 0.15
                or is_beat
                or (hue_change > 30 and time_since_update > 0.08)
            )

            if should_update:
                last_update = now
                last_hue = hue

                if color_mode == "complementary":
                    await controller.set_complementary(hue, sat, bright)
                else:
                    await controller.set_all_hsv(hue, sat, bright)

            await asyncio.sleep(1 / UPDATE_RATE)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\n\nError in visualization: {e}")
        import traceback

        traceback.print_exc()
    finally:
        analyzer.stop()
        if not stop_event:
            await controller.disconnect()
        print(f"\nDone! Total beats detected: {mapper.beat_count}")


async def run_movie_visualization(
    controller, mapper, analyzer=None, stop_event=None
):
    """Movie mode visualization loop - dual bulb colors from screen capture.

    Args:
        controller: BulbController instance
        mapper: MovieMapper instance
        analyzer: AudioAnalyzer instance (optional - if None, no audio influence)
        stop_event: asyncio.Event to signal stop (optional)
    """
    use_audio = analyzer is not None and mapper.audio_blend > 0

    try:
        print("\nStarting movie mode...")
        print(f"Target window: {mapper.window_name}")
        if use_audio:
            print(f"Audio blend: {mapper.audio_blend:.1f}")
        else:
            print("Audio: disabled (screen colors only)")

        # Check if window exists
        bounds = find_window(mapper.window_name)
        if not bounds:
            print(f"\nWarning: Window '{mapper.window_name}' not found!")
            print("Make sure the app is open and visible. Will retry continuously...")
        else:
            print(f"Window found: {bounds['width']}x{bounds['height']}")

        print("Press Ctrl+C to stop.\n")

        if use_audio:
            analyzer.start()

        # Get bulb names for left/right assignment
        bulb_names = list(controller.bulbs.keys())
        if len(bulb_names) < 2:
            print("Warning: Movie mode works best with 2 bulbs. Using same color for all.")
            left_bulb = bulb_names[0] if bulb_names else None
            right_bulb = bulb_names[0] if bulb_names else None
        else:
            left_bulb = bulb_names[0]
            right_bulb = bulb_names[1]
            print(f"Left bulb: {left_bulb}, Right bulb: {right_bulb}")

        last_update = 0

        while True:
            if stop_event and stop_event.is_set():
                break

            # Get audio bands if using audio, otherwise None
            bands = analyzer.get_frequency_bands() if use_audio else None
            left_hsv, right_hsv, is_beat = mapper.map_to_dual_hsv(bands)

            window_status = "📺" if mapper.window_bounds else "⚠️"

            if use_audio:
                beat_indicator = " BEAT!" if is_beat else ""
                print(
                    f"\r{window_status} L[H:{left_hsv[0]:3d} S:{left_hsv[1]:3d} B:{left_hsv[2]:3d}] "
                    f"R[H:{right_hsv[0]:3d} S:{right_hsv[1]:3d} B:{right_hsv[2]:3d}] "
                    f"Beats: {mapper.beat_count:4d}{beat_indicator:6s}",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\r{window_status} L[H:{left_hsv[0]:3d} S:{left_hsv[1]:3d} B:{left_hsv[2]:3d}] "
                    f"R[H:{right_hsv[0]:3d} S:{right_hsv[1]:3d} B:{right_hsv[2]:3d}]",
                    end="",
                    flush=True,
                )

            now = time.time()
            time_since_update = now - last_update

            # Update more frequently for smooth color transitions
            if time_since_update > 0.1 or is_beat:
                last_update = now

                # Set left and right bulbs with different colors
                if left_bulb:
                    await controller.set_hsv(
                        left_bulb, left_hsv[0], left_hsv[1], left_hsv[2]
                    )
                if right_bulb and right_bulb != left_bulb:
                    await controller.set_hsv(
                        right_bulb, right_hsv[0], right_hsv[1], right_hsv[2]
                    )

            await asyncio.sleep(1 / UPDATE_RATE)

    except KeyboardInterrupt:
        print("\n\nStopping movie mode...")
    except Exception as e:
        print(f"\n\nError in movie visualization: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if use_audio:
            analyzer.stop()
        if not stop_event:
            await controller.disconnect()
        if use_audio:
            print(f"\nDone! Total beats detected: {mapper.beat_count}")
        else:
            print("\nDone!")


class ConsoleCompleter:
    """Tab completion for console mode."""

    def __init__(self):
        # Main commands
        self.commands = [
            "reactive", "album", "calm", "rave", "fixed", "movie",
            "windows", "white", "off", "test", "devices", "status",
            "stop", "clear", "exit", "quit"
        ]

        # Options for visualization modes (reactive, album, calm, rave, fixed)
        self.viz_options = ["complementary", "device", "sens", "max"]

        # Options for movie mode
        self.movie_options = ["window", "title", "audio", "device", "sens", "max"]

        # White mode temps
        self.white_temps = ["cool2", "cool1", "neutral", "warm1", "warm2"]

        # Common hue values for fixed mode
        self.hue_presets = ["0", "30", "60", "120", "180", "240", "270", "300"]

        self.matches = []

    def complete(self, text, state):
        """Return the state'th completion for text."""
        if state == 0:
            # Get the full line buffer
            line = readline.get_line_buffer()
            begin = readline.get_begidx()

            # Parse what's been typed so far
            parts = line[:begin].split()

            if not parts:
                # Completing first word (command)
                self.matches = [c for c in self.commands if c.startswith(text.lower())]
            else:
                cmd = parts[0].lower()

                if cmd in ["reactive", "album", "calm", "rave"]:
                    # Visualization mode options
                    self.matches = [o for o in self.viz_options if o.startswith(text.lower())]

                elif cmd == "fixed":
                    # Fixed mode: first arg is hue, then options
                    if len(parts) == 1:
                        # Completing hue value
                        self.matches = [h for h in self.hue_presets if h.startswith(text)]
                    else:
                        self.matches = [o for o in self.viz_options if o.startswith(text.lower())]

                elif cmd == "movie":
                    self.matches = [o for o in self.movie_options if o.startswith(text.lower())]

                elif cmd == "white":
                    if len(parts) == 1:
                        # Completing temperature
                        self.matches = [t for t in self.white_temps if t.startswith(text.lower())]
                    else:
                        # Completing brightness - no suggestions
                        self.matches = []

                elif cmd == "windows":
                    # Could suggest app names, but leave empty for now
                    self.matches = []

                else:
                    self.matches = []

        try:
            return self.matches[state]
        except IndexError:
            return None


def setup_readline_completion():
    """Set up tab completion for console mode. Returns the completer and readline type."""
    completer = ConsoleCompleter()

    # Detect readline type
    is_gnureadline = 'gnureadline' in sys.modules
    is_libedit = not is_gnureadline and readline.__doc__ and 'libedit' in readline.__doc__

    # Set completer
    readline.set_completer(completer.complete)
    readline.set_completer_delims(" \t\n")

    if is_libedit:
        # macOS libedit - these bindings should work
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        # GNU readline (including gnureadline)
        readline.parse_and_bind("tab: complete")
        readline.parse_and_bind("set show-all-if-ambiguous on")

    return completer, is_libedit, is_gnureadline


def console_input(prompt, completer, is_libedit):
    """Get input with readline completion support."""
    # Re-apply completer settings before each input
    readline.set_completer(completer.complete)
    readline.set_completer_delims(" \t\n")

    if is_libedit:
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

    return input(prompt)


async def run_console(controller, default_device=None):
    """Interactive console mode - connect once, run multiple commands."""
    # Set up readline for command history and tab completion
    completer, is_libedit, is_gnureadline = setup_readline_completion()

    print("\nConsole mode - type commands to control lights")
    if is_libedit and not is_gnureadline:
        print("(Tab completion may be limited. Install gnureadline for better support: pip install gnureadline)")
    print("=" * 50)
    print("Commands:")
    print("  reactive [complementary] [device N] [sens X]  - Full spectrum mode")
    print("  album [complementary] [device N] [sens X]     - Album artwork colors")
    print("  calm [complementary] [device N] [sens X]      - Cool chill colors")
    print("  rave [complementary] [device N] [sens X]      - Party mode")
    print("  fixed HUE [complementary] [device N] [sens X] - Fixed hue (e.g. fixed 240)")
    print("  movie [window APP] [title X] [audio X]        - Screen colors (prompts for window)")
    print("  windows [APP]                                 - List windows for app")
    print("  white TEMP BRIGHTNESS                         - Static white (e.g. warm2 80)")
    print("  off                                           - Turn off all lights")
    print("  test                                          - Cycle through colors")
    print("  devices                                       - List audio input devices")
    print("  status                                        - Show current task status")
    print("  stop                                          - Stop current visualization")
    print("  clear                                         - Clear screen (or press Ctrl+L)")
    print("  exit / quit                                   - Disconnect and exit")
    print("\nOptions:")
    print("  complementary  - Opposite colors on each bulb")
    print("  device N       - Use audio device N (see 'devices' command)")
    print("  sens X         - Beat sensitivity (0.5=less, 2.0=more, default=1.0)")
    print("  max N          - Max brightness (1-100, default=100)")
    print("  window APP     - Target app for movie mode (default: Google Chrome)")
    print("  title X        - Match window by title (partial match, skips picker)")
    print("  audio X        - Audio blend for movie mode (0=none, 1=full, default=0.3)")
    print("\nTips:")
    print("  - Tab to autocomplete commands and options (cycles through matches)")
    print("  - Up/Down arrows to cycle through command history")
    print("  - Ctrl+C to stop current visualization")
    print("  - Ctrl+L to clear screen")
    print("=" * 50)
    print(f"Connected to: {', '.join(controller.bulbs.keys())}")
    print()

    current_task = None
    stop_event = None

    while True:
        try:
            # Print newline if visualization was running (to avoid prompt on same line)
            if current_task and not current_task.done():
                print()  # Move to new line

            # Prompt - use wrapper to ensure readline completion works
            cmd = await asyncio.get_event_loop().run_in_executor(
                None, console_input, "dj> ", completer, is_libedit
            )
            cmd = cmd.strip()

            if not cmd:
                continue

            # Parse command
            parts = shlex.split(cmd.lower())
            command = parts[0] if parts else ""

            # Exit commands
            if command in ["exit", "quit"]:
                print("Disconnecting...")
                break

            # Stop visualization
            elif command == "stop":
                if current_task and not current_task.done():
                    print("\nStopping visualization...")
                    if stop_event:
                        stop_event.set()
                    await asyncio.sleep(0.3)
                    print("Stopped.\n")
                else:
                    print("No visualization running")

            # Turn off lights
            elif command == "off":
                # Stop any running visualization first
                if current_task and not current_task.done():
                    if stop_event:
                        stop_event.set()
                    await asyncio.sleep(0.2)
                await controller.turn_off_all()

            # Clear screen
            elif command == "clear":
                os.system("clear" if os.name != "nt" else "cls")

            # List devices
            elif command in ["devices", "list"]:
                list_audio_devices()

            # List windows for an app
            elif command == "windows":
                # Get app name from args or default to Chrome
                if len(parts) > 1:
                    # Preserve original case for app name
                    original_parts = shlex.split(cmd)
                    app_name = original_parts[1]
                else:
                    app_name = "Google Chrome"

                windows = list_app_windows(app_name)
                if not windows:
                    print(f"\nNo windows found for '{app_name}'")
                    print("Make sure the app is open and has visible windows.")
                else:
                    print(f"\nWindows for '{app_name}':")
                    print("-" * 60)
                    for i, win in enumerate(sorted(windows, key=lambda w: w['width'] * w['height'], reverse=True)):
                        print(f"  [{i}] {win['title']}")
                        print(f"      Size: {win['width']}x{win['height']}")
                    print("-" * 60)
                    print(f"Use 'movie index N' to select a window (e.g. 'movie index 1')\n")

            # Status
            elif command == "status":
                if current_task and not current_task.done():
                    print(f"Task running: {current_task}")
                    print(f"Task done: {current_task.done()}")
                elif current_task:
                    print(f"Task completed: {current_task}")
                    try:
                        result = current_task.result()
                        print(f"Result: {result}")
                    except Exception as e:
                        print(f"Task failed with error: {e}")
                else:
                    print("No task running")

            # Test mode
            elif command == "test":
                print("\nTest mode: cycling through colors...")
                for hue in range(0, 360, 30):
                    print(f"  Hue: {hue}")
                    await controller.set_all_hsv(hue, 100, 100)
                    await asyncio.sleep(1)
                print("Test complete!\n")

            # White mode
            elif command == "white":
                if len(parts) < 3:
                    print("Usage: white <temp> <brightness>")
                    print("Example: white warm2 80")
                    continue
                color_temp = parts[1]
                try:
                    brightness = int(parts[2])
                    await run_white_mode(
                        controller, brightness, color_temp, disconnect=False
                    )
                except ValueError:
                    print("Brightness must be a number (1-100)")

            # Visualization modes
            elif command in ["reactive", "album", "calm", "rave", "fixed"]:
                # Stop current visualization if running
                if current_task and not current_task.done():
                    print("\nStopping current visualization...")
                    if stop_event:
                        stop_event.set()
                    await asyncio.sleep(0.3)

                # Parse options
                color_mode = "sync"
                device = default_device
                sensitivity = 1.0
                max_brightness = 100
                fixed_hue = 240  # Default blue for fixed mode

                # For fixed mode, first argument is the hue
                if command == "fixed":
                    if len(parts) < 2:
                        print("Usage: fixed HUE [complementary] [device N] [sens X] [max N]")
                        print("Example: fixed 240 complementary device 2 sens 1.5 max 50")
                        continue
                    try:
                        fixed_hue = int(parts[1]) % 360
                        parts_start = 2
                    except ValueError:
                        print(f"Invalid hue value: {parts[1]} (must be 0-360)")
                        continue
                else:
                    parts_start = 1

                # Parse remaining options
                i = parts_start
                while i < len(parts):
                    part = parts[i]
                    if part == "complementary":
                        color_mode = "complementary"
                        i += 1
                    elif part == "device" and i + 1 < len(parts):
                        try:
                            device = int(parts[i + 1])
                            i += 2
                        except ValueError:
                            print(f"Invalid device number: {parts[i + 1]}")
                            i += 2
                    elif part == "sens" and i + 1 < len(parts):
                        try:
                            sensitivity = float(parts[i + 1])
                            sensitivity = max(0.1, min(3.0, sensitivity))
                            i += 2
                        except ValueError:
                            print(f"Invalid sensitivity value: {parts[i + 1]} (0.5-2.0 recommended)")
                            i += 2
                    elif part == "max" and i + 1 < len(parts):
                        try:
                            max_brightness = int(parts[i + 1])
                            max_brightness = max(1, min(100, max_brightness))
                            i += 2
                        except ValueError:
                            print(f"Invalid max brightness: {parts[i + 1]} (must be 1-100)")
                            i += 2
                    else:
                        i += 1

                # Select mapper
                mappers = {
                    "reactive": lambda: ReactiveMapper(sensitivity=sensitivity, max_brightness=max_brightness),
                    "album": lambda: AlbumReactiveMapper(sensitivity=sensitivity, max_brightness=max_brightness),
                    "calm": lambda: CalmMapper(sensitivity=sensitivity, max_brightness=max_brightness),
                    "rave": lambda: RaveMapper(sensitivity=sensitivity, max_brightness=max_brightness),
                    "fixed": lambda: FixedHueMapper(hue=fixed_hue, sensitivity=sensitivity, max_brightness=max_brightness),
                }
                mapper = mappers[command]()
                analyzer = AudioAnalyzer(device=device)

                # Create stop event and run in background
                stop_event = asyncio.Event()
                current_task = asyncio.create_task(
                    run_visualization(
                        controller, analyzer, mapper, color_mode, stop_event
                    )
                )

                # Add callback to catch task exceptions
                def task_done_callback(task):
                    try:
                        task.result()
                    except Exception as e:
                        print(f"\n\nVisualization error: {e}")
                        import traceback

                        traceback.print_exc()

                current_task.add_done_callback(task_done_callback)

                # Give task a moment to start
                await asyncio.sleep(0.1)

            # Movie mode (screen capture)
            elif command == "movie":
                # Stop current visualization if running
                if current_task and not current_task.done():
                    print("\nStopping current visualization...")
                    if stop_event:
                        stop_event.set()
                    await asyncio.sleep(0.3)

                # Parse options
                device = default_device
                sensitivity = 1.0
                audio_blend = 0.3
                max_brightness = 100
                window_name = "Google Chrome"
                window_title = None  # Will be set by picker or 'title' option

                # Parse options
                i = 1
                while i < len(parts):
                    part = parts[i]
                    if part == "window" and i + 1 < len(parts):
                        # Window name (app name) - preserve original case from user input
                        original_parts = shlex.split(cmd)  # Re-parse without lowercase
                        window_name = original_parts[i + 1]
                        i += 2
                    elif part == "title" and i + 1 < len(parts):
                        # Window title to match - preserve original case
                        original_parts = shlex.split(cmd)
                        window_title = original_parts[i + 1]
                        i += 2
                    elif part == "audio" and i + 1 < len(parts):
                        try:
                            audio_blend = float(parts[i + 1])
                            audio_blend = max(0.0, min(1.0, audio_blend))
                            i += 2
                        except ValueError:
                            print(f"Invalid audio blend value: {parts[i + 1]} (0-1)")
                            i += 2
                    elif part == "device" and i + 1 < len(parts):
                        try:
                            device = int(parts[i + 1])
                            i += 2
                        except ValueError:
                            print(f"Invalid device number: {parts[i + 1]}")
                            i += 2
                    elif part == "sens" and i + 1 < len(parts):
                        try:
                            sensitivity = float(parts[i + 1])
                            sensitivity = max(0.1, min(3.0, sensitivity))
                            i += 2
                        except ValueError:
                            print(f"Invalid sensitivity value: {parts[i + 1]}")
                            i += 2
                    elif part == "max" and i + 1 < len(parts):
                        try:
                            max_brightness = int(parts[i + 1])
                            max_brightness = max(1, min(100, max_brightness))
                            i += 2
                        except ValueError:
                            print(f"Invalid max brightness: {parts[i + 1]} (must be 1-100)")
                            i += 2
                    else:
                        i += 1

                # If no title specified, prompt user to pick a window
                if window_title is None:
                    window_title = await pick_window_interactive(window_name)
                    if window_title is None:
                        continue  # User cancelled

                # Create movie mapper
                mapper = MovieMapper(
                    sensitivity=sensitivity,
                    audio_blend=audio_blend,
                    window_name=window_name,
                    window_title=window_title,
                    max_brightness=max_brightness
                )

                # Only create analyzer if audio is enabled
                analyzer = AudioAnalyzer(device=device) if audio_blend > 0 else None

                # Create stop event and run in background
                stop_event = asyncio.Event()
                current_task = asyncio.create_task(
                    run_movie_visualization(
                        controller, mapper, analyzer, stop_event
                    )
                )

                # Add callback to catch task exceptions
                def task_done_callback(task):
                    try:
                        task.result()
                    except Exception as e:
                        print(f"\n\nMovie mode error: {e}")
                        import traceback

                        traceback.print_exc()

                current_task.add_done_callback(task_done_callback)

                # Give task a moment to start
                await asyncio.sleep(0.1)

            else:
                print(f"Unknown command: {command}")
                print(
                    "Try: reactive, album, calm, rave, fixed, movie, windows, white, off, test, devices, status, stop, clear, exit"
                )

        except EOFError:
            print("\nExiting...")
            break
        except KeyboardInterrupt:
            # Stop current visualization on Ctrl+C
            if current_task and not current_task.done():
                print("\n\nStopping visualization...")
                if stop_event:
                    stop_event.set()
                await asyncio.sleep(0.3)  # Give it time to stop
                print("Stopped. Type a new command or 'exit' to quit.\n")
            else:
                print("\n(Type 'exit' to quit)\n")
            continue

    # Cleanup
    if current_task and not current_task.done():
        if stop_event:
            stop_event.set()
        await asyncio.sleep(0.2)

    await controller.disconnect()
    print("Goodbye!")


async def main():
    parser = argparse.ArgumentParser(
        description="DJ Lights - Music visualization for Kasa bulbs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  reactive      Full spectrum, beat-driven color changes (default)
  calm          Subtle warm colors, slow transitions (for studying)
  rave          Aggressive, fast, strobe effects on beats
  white         Static warm/cool white light

Examples:
  %(prog)s --mode reactive --device 2
  %(prog)s --mode calm
  %(prog)s --mode white --brightness 80 --color-temp warm1
  %(prog)s --mode rave --color-mode complementary

Environment:
  Set BULBS='left:192.168.0.103,right:192.168.0.104' before running.
""",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio input devices"
    )
    parser.add_argument("--device", type=int, help="Audio input device index")
    parser.add_argument(
        "--mode",
        choices=["reactive", "album", "calm", "rave", "white", "fixed", "movie"],
        default="reactive",
        help="Visualization mode",
    )
    parser.add_argument(
        "--color-mode",
        choices=["sync", "complementary"],
        default="sync",
        help="Color mode: sync (same) or complementary (opposite)",
    )
    parser.add_argument(
        "--brightness", type=int, default=80, help="Brightness for white mode (1-100)"
    )
    parser.add_argument(
        "--color-temp",
        choices=["cool2", "cool1", "neutral", "warm1", "warm2"],
        default="warm1",
        help="Color temperature for white mode (cool2=daylight, warm2=candlelight)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: cycle through colors"
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Interactive console mode - connect once, run multiple commands",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Beat detection sensitivity (0.5=less sensitive, 2.0=more sensitive, default=1.0)",
    )
    parser.add_argument(
        "--hue",
        type=int,
        default=240,
        help="Fixed hue value for fixed mode (0-360, default=240 for blue)",
    )
    parser.add_argument(
        "--audio-blend",
        type=float,
        default=0.3,
        help="Audio influence on brightness for movie mode (0=none, 1=full, default=0.3)",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="Google Chrome",
        help="Target app name for movie mode (default='Google Chrome')",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default=None,
        help="Match window by title (partial match). If not set, prompts for selection.",
    )
    parser.add_argument(
        "--list-windows",
        type=str,
        nargs="?",
        const="Google Chrome",
        help="List windows for an app (default: Google Chrome)",
    )
    parser.add_argument(
        "--off",
        action="store_true",
        help="Turn off all lights and exit",
    )
    parser.add_argument(
        "--max-brightness",
        type=int,
        default=100,
        help="Maximum brightness limit (1-100, default=100). Useful when roommate is sleeping!",
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.list_windows:
        app_name = args.list_windows
        windows = list_app_windows(app_name)
        if not windows:
            print(f"\nNo windows found for '{app_name}'")
            print("Make sure the app is open and has visible windows.")
        else:
            print(f"\nWindows for '{app_name}':")
            print("-" * 60)
            for i, win in enumerate(sorted(windows, key=lambda w: w['width'] * w['height'], reverse=True)):
                print(f"  [{i}] {win['title']}")
                print(f"      Size: {win['width']}x{win['height']}")
            print("-" * 60)
            print(f"\nUse --window-title 'PARTIAL_TITLE' to match a window")
            print(f"Example: --window-title 'Netflix'")
        return

    if args.off:
        print("DJ Lights - Turning off")
        print("=" * 50)
        bulb_ips = parse_bulbs_env()
        if not bulb_ips:
            print("\nNo bulbs configured. Set BULBS environment variable.")
            return
        controller = BulbController(bulb_ips)
        await controller.connect()
        await controller.turn_off_all()
        await controller.disconnect()
        return

    print("DJ Lights - Real-time music visualization")
    print("=" * 50)

    # Parse bulbs from environment
    bulb_ips = parse_bulbs_env()
    if not bulb_ips:
        print("\nNo bulbs configured. Set BULBS environment variable.")
        return

    print(f"Bulbs: {bulb_ips}")

    # Validate max brightness
    max_brightness = max(1, min(100, args.max_brightness))
    if max_brightness < 100:
        print(f"Max brightness: {max_brightness}%")

    controller = BulbController(bulb_ips)
    await controller.connect()
    await controller.turn_on_all()

    if args.console:
        await run_console(controller, default_device=args.device)
        return

    if args.test:
        print("\nTest mode: cycling through colors...")
        for hue in range(0, 360, 30):
            print(f"  Hue: {hue}")
            await controller.set_all_hsv(hue, 100, 100)
            await asyncio.sleep(1)
        await controller.disconnect()
        print("Test complete!")
        return

    if args.mode == "white":
        await run_white_mode(controller, args.brightness, args.color_temp)
        return

    # Validate sensitivity range
    sensitivity = max(0.1, min(3.0, args.sensitivity))
    if sensitivity != args.sensitivity:
        print(f"Warning: Sensitivity clamped to valid range (0.1-3.0): {sensitivity}")

    # Movie mode uses a different visualization function
    if args.mode == "movie":
        audio_blend = max(0.0, min(1.0, args.audio_blend))

        # If no window title specified, prompt for selection
        window_title = args.window_title
        if window_title is None:
            window_title = await pick_window_interactive(args.window)
            if window_title is None:
                print("No window selected. Exiting.")
                await controller.disconnect()
                return

        mapper = MovieMapper(
            sensitivity=sensitivity,
            audio_blend=audio_blend,
            window_name=args.window,
            window_title=window_title,
            max_brightness=max_brightness
        )
        # Only create analyzer if audio is enabled
        analyzer = AudioAnalyzer(device=args.device) if audio_blend > 0 else None
        await run_movie_visualization(controller, mapper, analyzer)
        return

    # Select mapper based on mode
    mappers = {
        "reactive": lambda: ReactiveMapper(sensitivity=sensitivity, max_brightness=max_brightness),
        "album": lambda: AlbumReactiveMapper(sensitivity=sensitivity, max_brightness=max_brightness),
        "calm": lambda: CalmMapper(sensitivity=sensitivity, max_brightness=max_brightness),
        "rave": lambda: RaveMapper(sensitivity=sensitivity, max_brightness=max_brightness),
        "fixed": lambda: FixedHueMapper(hue=args.hue, sensitivity=sensitivity, max_brightness=max_brightness),
    }
    mapper = mappers[args.mode]()

    analyzer = AudioAnalyzer(device=args.device)
    await run_visualization(controller, analyzer, mapper, args.color_mode)


if __name__ == "__main__":
    asyncio.run(main())

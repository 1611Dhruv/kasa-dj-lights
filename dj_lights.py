#!/usr/bin/env python3
"""
DJ Lights - Real-time music visualization for Kasa smart bulbs.
Captures audio, analyzes frequencies via FFT, and maps them to colors.

Environment Variables:
    BULBS - Comma or colon separated list of bulb IPs
           Format: "name1:ip1,name2:ip2" or just "ip1,ip2"
           Example: "left:192.168.0.103,right:192.168.0.104"
"""

import argparse
import asyncio
import os
import sys
import time
from collections import deque

import argcomplete
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from kasa import Discover
from kasa.module import Module
from scipy.fft import rfft, rfftfreq

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

    def __init__(self):
        self.peak_bass = 0.1
        self.peak_mid = 0.1
        self.peak_treble = 0.1
        self.peak_energy = 0.1
        self.decay = 0.995
        self.last_bass = 0
        self.beat_count = 0

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
        is_beat = bass_delta > 0.3 and norm_bass > 0.5
        self.last_bass = norm_bass * 0.7 + self.last_bass * 0.3
        if is_beat:
            self.beat_count += 1
        return is_beat


class ReactiveMapper(ColorMapper):
    """Default reactive mode - full spectrum, beat-driven hue shifts."""

    def __init__(self):
        super().__init__()
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

        return hue, min(100, saturation), min(100, brightness), is_beat


class CalmMapper(ColorMapper):
    """Calm mode - slow, gradual, cool chill colors (blues, greens, cyans)."""

    def __init__(self):
        super().__init__()
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

        return int(self.hue) % 360, int(self.saturation), int(self.brightness), False


class RaveMapper(ColorMapper):
    """Rave mode - aggressive, fast, full spectrum, strobe on beats."""

    def __init__(self):
        super().__init__()
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

        return self.hue, saturation, min(100, brightness), is_beat


class BulbController:
    """Controls Kasa smart bulbs."""

    def __init__(self, ips: dict):
        self.ips = ips
        self.bulbs = {}
        self.last_hsv = {}

    async def connect(self):
        for name, ip in self.ips.items():
            try:
                bulb = await Discover.discover_single(ip)
                await bulb.update()
                self.bulbs[name] = bulb
                self.last_hsv[name] = (0, 0, 0)
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

    async def set_hsv(self, name: str, hue: int, saturation: int, brightness: int):
        if name not in self.bulbs:
            return

        last = self.last_hsv[name]
        if (
            abs(hue - last[0]) < 5
            and abs(saturation - last[1]) < 5
            and abs(brightness - last[2]) < 5
        ):
            return

        try:
            light = self.bulbs[name].modules[Module.Light]
            await light.set_hsv(hue, saturation, brightness)
            self.last_hsv[name] = (hue, saturation, brightness)
        except Exception as e:
            print(f"Error setting {name}: {e}", file=sys.stderr)

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


async def run_white_mode(controller, brightness: int, color_temp: str):
    """Static warm white light mode."""
    # Color temperatures from cool to warm
    temps = {
        "cool2": (210, 25),   # Daylight blue
        "cool1": (180, 20),   # Cool white
        "neutral": (60, 15),  # Neutral white
        "warm1": (35, 30),    # Warm white
        "warm2": (25, 40),    # Candlelight
    }
    hue, sat = temps.get(color_temp, temps["neutral"])

    print(f"\nWhite mode: {color_temp}, brightness {brightness}%")
    await controller.set_all_hsv(hue, sat, brightness)
    await controller.disconnect()
    print("Done!")


async def run_visualization(controller, analyzer, mapper, color_mode: str):
    """Main visualization loop."""
    print("\nStarting audio capture...")
    print("(Use --list-devices to see available inputs)")
    print("(Set BlackHole as input to capture system audio)")
    print(f"\nMode: {mapper.__class__.__name__}")
    print("Press Ctrl+C to stop.\n")

    analyzer.start()

    last_update = 0
    last_hue = 0

    try:
        while True:
            bands = analyzer.get_frequency_bands()
            hue, sat, bright, is_beat = mapper.map_to_hsv(bands)

            beat_indicator = " BEAT!" if is_beat else ""
            print(
                f"\rHue: {hue:3d} | Sat: {sat:3d} | Bright: {bright:3d} | "
                f"Beats: {mapper.beat_count:4d}{beat_indicator:6s}",
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
    finally:
        analyzer.stop()
        await controller.disconnect()
        print(f"Done! Total beats detected: {mapper.beat_count}")


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
        choices=["reactive", "calm", "rave", "white"],
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
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    print("DJ Lights - Real-time music visualization")
    print("=" * 50)

    # Parse bulbs from environment
    bulb_ips = parse_bulbs_env()
    if not bulb_ips:
        print("\nNo bulbs configured. Set BULBS environment variable.")
        return

    print(f"Bulbs: {bulb_ips}")

    controller = BulbController(bulb_ips)
    await controller.connect()
    await controller.turn_on_all()

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

    # Select mapper based on mode
    mappers = {
        "reactive": ReactiveMapper,
        "calm": CalmMapper,
        "rave": RaveMapper,
    }
    mapper = mappers[args.mode]()

    analyzer = AudioAnalyzer(device=args.device)
    await run_visualization(controller, analyzer, mapper, args.color_mode)


if __name__ == "__main__":
    asyncio.run(main())

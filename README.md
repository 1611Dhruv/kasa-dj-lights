# DJ Lights

Real-time music visualization for Kasa smart bulbs. Captures system audio, analyzes frequencies via FFT, and maps them to colors.

## Setup

### 1. Install dependencies

```bash
pip install python-kasa sounddevice numpy scipy python-dotenv argcomplete pillow scikit-learn mss pyobjc-framework-Quartz

# Optional: For better tab completion on macOS
pip install gnureadline
```

### 2. Install BlackHole (for system audio capture)

```bash
brew install blackhole-2ch
```

Then set up a Multi-Output Device in macOS Audio MIDI Setup to route audio to both speakers and BlackHole.

### 3. Discover your bulbs

```bash
kasa discover
```

Or target a specific subnet:
```bash
kasa discover --target 192.168.0.255
```

This will output something like:
```
Discovering devices on 192.168.0.255 for 3 seconds
== Bulb Left - KL135(US) ==
    Host: 192.168.0.103
    ...
== Bulb Right - KL135(US) ==
    Host: 192.168.0.104
    ...
```

### 4. Configure bulbs

Create a `.env` file or export the environment variable with the IPs from discovery:

```bash
export BULBS='left:192.168.0.103,right:192.168.0.104'
```

Or just IPs (auto-named bulb1, bulb2...):
```bash
export BULBS='192.168.0.103,192.168.0.104'
```

### 5. (Optional) Enable tab completion

Add to `~/.zshrc`:
```bash
autoload -U bashcompinit
bashcompinit
eval "$(register-python-argcomplete dj_lights.py)"
```

## Modes

| Mode | Description |
|------|-------------|
| `reactive` | Full spectrum, beat-driven color changes (default) |
| `album` | **Smart adaptive mode** - uses album artwork colors, adapts to song energy |
| `calm` | Cool colors (blues/greens/cyans), slow gradual transitions |
| `rave` | Aggressive, fast cycling, strobe effects on beats |
| `fixed` | Fixed hue - only varies saturation/brightness with music (you choose the color) |
| `movie` | **Ambilight-style** - captures screen colors from a window (Chrome by default) |
| `white` | Static white light with adjustable temperature |

**Album Mode** (Adaptive):
- Extracts colors from Apple Music album artwork (including low-sat colors like whites/grays)
- **Intelligently adapts** to song vibe:
  - 😌 **Chill songs**: Smooth transitions, subtle changes, stays on colors longer
  - 🎵 **Medium energy**: Balanced responsiveness
  - 🔊 **Hype songs**: Dramatic reactions, rapid color changes, beatdrop detection
- Brightness pulses with beats (scaled to song energy)
- Beatdrops trigger intense effects (only on high-energy songs)
- Auto-updates palette when track changes

**Movie Mode** (Screen Capture):
- Captures colors from left/right halves of a target window (like Philips Ambilight)
- Left bulb shows left half colors, right bulb shows right half colors
- **Interactive picker** - prompts you to select which window to capture
- **Title matching** - use `title Netflix` or `--window-title Netflix` to skip picker and match by title
- Title stays consistent even when window order changes!
- **Audio is optional** - set `audio 0` for pure screen colors (no audio device needed)
- Note: First run requires macOS Screen Recording permission

## Console Mode (Recommended)

Connect once, run multiple commands without reconnecting:

```bash
python dj_lights.py --console --device 2
```

Then use simple commands:
```
dj> reactive
dj> album                   # Album artwork colors!
dj> calm complementary
dj> fixed 240               # Fixed blue hue (240°)
dj> fixed 0 sens 1.5        # Fixed red hue with higher beat sensitivity
dj> rave sens 0.7           # Rave mode with lower beat sensitivity
dj> movie                   # Prompts to pick Chrome window, then captures it
dj> movie audio 0           # No audio needed - pure screen colors
dj> movie window Safari     # Pick from Safari windows instead
dj> movie title Netflix     # Skip picker, match window with "Netflix" in title
dj> windows                 # List Chrome windows
dj> windows Safari          # List Safari windows
dj> reactive max 50         # Reactive mode, capped at 50% brightness
dj> album max 40            # Album mode, capped at 40%
dj> movie max 30            # Movie mode, capped at 30%
dj> off                     # Turn off all lights
dj> stop                    # Stop visualization
dj> white warm2 80
dj> devices                 # List audio devices
dj> status                  # Check task status
dj> clear                   # Clear screen
dj> exit
```

**Why console mode?** Connects to bulbs once and keeps the connection alive. No more waiting for reconnections between commands!

**Tips:**
- **Tab** - Autocomplete commands and options (cycles through matches)
- **Up/Down arrows** - Cycle through command history
- **Ctrl+C** or `stop` - Stop the current visualization (stays in console)
- **Ctrl+L** or `clear` - Clear the screen
- `devices` - List audio inputs
- `status` - Check if visualization is running
- `exit` - Disconnect and quit

## Options Reference

| Option | Values | Description |
|--------|--------|-------------|
| `--console` | | Interactive console - connect once, run multiple commands |
| `--mode` | `reactive`, `album`, `calm`, `rave`, `fixed`, `movie`, `white` | Visualization mode |
| `--device` | number | Audio input device index |
| `--color-mode` | `sync`, `complementary` | Same color or opposite colors on bulbs |
| `--sensitivity` | 0.5-2.0 | Beat detection sensitivity (default: 1.0) |
| `--hue` | 0-360 | Hue value for fixed mode (default: 240 for blue) |
| `--window` | app name | Target app for movie mode (default: "Google Chrome") |
| `--window-title` | partial title | Match window by title (skips picker) |
| `--list-windows` | [app name] | List windows for an app (default: Chrome) |
| `--audio-blend` | 0-1 | Audio influence on brightness for movie mode (default: 0.3) |
| `--off` | | Turn off all lights and exit |
| `--max-brightness` | 1-100 | Maximum brightness limit (default: 100). Useful when roommate is sleeping! |
| `--brightness` | 1-100 | Brightness for white mode |
| `--color-temp` | `cool2`, `cool1`, `neutral`, `warm1`, `warm2` | Color temperature for white mode |
| `--list-devices` | | List available audio inputs |
| `--test` | | Cycle through colors to test bulbs |

## Color Temperatures

| Temp | Description |
|------|-------------|
| `cool2` | Daylight blue |
| `cool1` | Cool white |
| `neutral` | Neutral white |
| `warm1` | Warm white (default) |
| `warm2` | Candlelight |

## Tips

- Use `--list-devices` to find BlackHole's device index
- `complementary` mode looks great with two bulbs - opposite colors
- **Album mode** syncs lights to album artwork - perfect for immersive listening!
- **Movie mode** captures screen colors - perfect for movies/shows in Chrome
- **Fixed mode** keeps your favorite color - perfect when you want consistency
- **Beat sensitivity**: Increase (`sens 1.5`) to catch more beats, decrease (`sens 0.7`) for only strong beats
- Calm mode is perfect for lo-fi / ambient / study sessions
- Rave mode works best with EDM / high-energy tracks

---

## Quick Commands

```bash
# Discover bulbs
kasa discover --target 192.168.0.255

# List audio devices
python dj_lights.py --list-devices

# Console mode (RECOMMENDED - connect once, run multiple commands)
python dj_lights.py --console --device 2

# One-off commands (reconnects each time)
# Test bulbs
python dj_lights.py --test

# Reactive (default)
python dj_lights.py --device 2
python dj_lights.py --device 2 --color-mode complementary

# Album artwork colors (Apple Music)
python dj_lights.py --mode album --device 2

# Calm study mode
python dj_lights.py --mode calm --device 2

# Rave party
python dj_lights.py --mode rave --device 2
python dj_lights.py --mode rave --device 2 --color-mode complementary

# Fixed hue (your favorite color)
python dj_lights.py --mode fixed --hue 240 --device 2  # Blue
python dj_lights.py --mode fixed --hue 0 --device 2    # Red
python dj_lights.py --mode fixed --hue 120 --device 2  # Green

# Adjust beat sensitivity
python dj_lights.py --mode reactive --device 2 --sensitivity 1.5  # More beats
python dj_lights.py --mode album --device 2 --sensitivity 0.7     # Less beats

# Movie mode (screen capture)
python dj_lights.py --mode movie                                    # Prompts to pick window
python dj_lights.py --mode movie --window-title Netflix             # Match window with "Netflix"
python dj_lights.py --mode movie --audio-blend 0                    # Pure screen colors
python dj_lights.py --mode movie --device 2 --audio-blend 0.5       # With audio influence
python dj_lights.py --mode movie --window Safari                    # Pick from Safari windows
python dj_lights.py --list-windows                                  # List Chrome windows
python dj_lights.py --list-windows Safari                           # List Safari windows

# Turn off lights
python dj_lights.py --off                                           # Turn off and exit

# Limit brightness (roommate sleeping!)
python dj_lights.py --console --device 2 --max-brightness 40        # Cap at 40%
python dj_lights.py --mode reactive --device 2 --max-brightness 50  # Never exceed 50%

# White light
python dj_lights.py --mode white --color-temp warm2 --brightness 60
python dj_lights.py --mode white --brightness 80
python dj_lights.py --mode white --color-temp neutral --brightness 100
python dj_lights.py --mode white --color-temp cool2 --brightness 100
```

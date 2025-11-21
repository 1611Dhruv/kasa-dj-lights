# DJ Lights

Real-time music visualization for Kasa smart bulbs. Captures system audio, analyzes frequencies via FFT, and maps them to colors.

## Setup

### 1. Install dependencies

```bash
pip install python-kasa sounddevice numpy scipy python-dotenv argcomplete
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
| `calm` | Cool colors (blues/greens/cyans), slow gradual transitions |
| `rave` | Aggressive, fast cycling, strobe effects on beats |
| `white` | Static white light with adjustable temperature |

## Options Reference

| Option | Values | Description |
|--------|--------|-------------|
| `--mode` | `reactive`, `calm`, `rave`, `white` | Visualization mode |
| `--device` | number | Audio input device index |
| `--color-mode` | `sync`, `complementary` | Same color or opposite colors on bulbs |
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
- Calm mode is perfect for lo-fi / ambient / study sessions
- Rave mode works best with EDM / high-energy tracks

---

## Quick Commands

```bash
# Discover bulbs
kasa discover --target 192.168.0.255

# List audio devices
python dj_lights.py --list-devices

# Test bulbs
python dj_lights.py --test

# Reactive (default)
python dj_lights.py --device 2
python dj_lights.py --device 2 --color-mode complementary

# Calm study mode
python dj_lights.py --mode calm --device 2

# Rave party
python dj_lights.py --mode rave --device 2
python dj_lights.py --mode rave --device 2 --color-mode complementary

# White light
python dj_lights.py --mode white --color-temp warm2 --brightness 60
python dj_lights.py --mode white --brightness 80
python dj_lights.py --mode white --color-temp neutral --brightness 100
python dj_lights.py --mode white --color-temp cool2 --brightness 100
```

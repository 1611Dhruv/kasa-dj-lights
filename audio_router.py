#!/usr/bin/env python3
"""
Audio Router - Simple menu bar app to manage BlackHole audio routing.
Allows you to:
- Control system volume
- Create/toggle multi-output devices (speakers + BlackHole)
- Switch between audio outputs quickly
"""

import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import json
import re


def run_cmd(cmd, capture=True):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True, timeout=10
        )
        return result.stdout.strip() if capture else None
    except Exception as e:
        print(f"Command error: {e}")
        return None


def get_volume():
    """Get current system volume (0-100)."""
    result = run_cmd("osascript -e 'output volume of (get volume settings)'")
    try:
        return int(result) if result else 50
    except ValueError:
        return 50


def set_volume(vol):
    """Set system volume (0-100)."""
    vol = max(0, min(100, int(vol)))
    run_cmd(f"osascript -e 'set volume output volume {vol}'", capture=False)


def is_muted():
    """Check if audio is muted."""
    result = run_cmd("osascript -e 'output muted of (get volume settings)'")
    return result == "true"


def toggle_mute():
    """Toggle mute state."""
    muted = is_muted()
    run_cmd(f"osascript -e 'set volume output muted {str(not muted).lower()}'", capture=False)
    return not muted


def get_audio_devices():
    """Get list of audio output devices using system_profiler."""
    result = run_cmd("system_profiler SPAudioDataType -json")
    if not result:
        return []

    try:
        data = json.loads(result)
        devices = []
        audio_data = data.get("SPAudioDataType", [])

        for item in audio_data:
            items = item.get("_items", [])
            for device in items:
                name = device.get("_name", "Unknown")
                # Check if it's an output device
                if "coreaudio_output_source" in device or "Output" in str(device):
                    devices.append({
                        "name": name,
                        "type": "output"
                    })

        return devices
    except json.JSONDecodeError:
        return []


def get_current_output():
    """Get current audio output device name."""
    # Try using SwitchAudioSource if available
    result = run_cmd("SwitchAudioSource -c 2>/dev/null")
    if result:
        return result

    # Fallback to AppleScript
    result = run_cmd('''osascript -e 'tell application "System Events" to get name of current output device of (get volume settings)' 2>/dev/null''')
    return result if result else "Unknown"


def set_output_device(device_name):
    """Set output device using SwitchAudioSource (if available) or AppleScript."""
    # Try SwitchAudioSource first
    result = run_cmd(f'SwitchAudioSource -s "{device_name}" 2>/dev/null')
    if result is not None:
        return True

    # Fallback method using AppleScript (limited)
    script = f'''
    tell application "System Preferences"
        reveal anchor "output" of pane id "com.apple.preference.sound"
    end tell
    '''
    run_cmd(f"osascript -e '{script}'", capture=False)
    return False


def list_switchaudio_devices():
    """List devices using SwitchAudioSource."""
    result = run_cmd("SwitchAudioSource -a -t output 2>/dev/null")
    if result:
        return [d.strip() for d in result.split("\n") if d.strip()]
    return []


def check_dependencies():
    """Check if required tools are installed."""
    missing = []

    # Check for SwitchAudioSource
    result = run_cmd("which SwitchAudioSource 2>/dev/null")
    if not result:
        missing.append("SwitchAudioSource")

    return missing


def create_multi_output_device():
    """Open Audio MIDI Setup to create multi-output device (as fallback)."""
    run_cmd("open -a 'Audio MIDI Setup'", capture=False)


class AudioRouterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Router")
        self.root.geometry("400x500")
        self.root.resizable(True, True)

        # Set minimum size
        self.root.minsize(350, 400)

        # Check dependencies
        missing = check_dependencies()
        if missing:
            self.show_dependency_warning(missing)

        self.create_widgets()
        self.update_device_list()
        self.update_volume_display()

        # Auto-refresh
        self.auto_refresh()

    def show_dependency_warning(self, missing):
        """Show warning about missing dependencies."""
        msg = "For full functionality, install:\n\n"
        if "SwitchAudioSource" in missing:
            msg += "• SwitchAudioSource:\n  brew install switchaudio-osx\n\n"
        msg += "Some features may be limited without these tools."
        messagebox.showwarning("Missing Dependencies", msg)

    def create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === VOLUME SECTION ===
        vol_frame = ttk.LabelFrame(main_frame, text="Volume", padding="10")
        vol_frame.pack(fill=tk.X, pady=(0, 10))

        # Volume slider
        self.volume_var = tk.IntVar(value=get_volume())
        self.volume_slider = ttk.Scale(
            vol_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.volume_var,
            command=self.on_volume_change
        )
        self.volume_slider.pack(fill=tk.X, pady=(0, 5))

        # Volume label and mute button
        vol_controls = ttk.Frame(vol_frame)
        vol_controls.pack(fill=tk.X)

        self.volume_label = ttk.Label(vol_controls, text="50%")
        self.volume_label.pack(side=tk.LEFT)

        self.mute_btn = ttk.Button(vol_controls, text="🔇 Mute", command=self.on_mute)
        self.mute_btn.pack(side=tk.RIGHT)

        # === OUTPUT DEVICE SECTION ===
        device_frame = ttk.LabelFrame(main_frame, text="Output Device", padding="10")
        device_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Current device
        current_frame = ttk.Frame(device_frame)
        current_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(current_frame, text="Current:").pack(side=tk.LEFT)
        self.current_device_label = ttk.Label(current_frame, text="Loading...", font=("", 11, "bold"))
        self.current_device_label.pack(side=tk.LEFT, padx=(5, 0))

        # Device list
        ttk.Label(device_frame, text="Available Devices:").pack(anchor=tk.W)

        # Listbox with scrollbar
        list_frame = ttk.Frame(device_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.device_listbox = tk.Listbox(list_frame, height=8, font=("", 11))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.device_listbox.yview)
        self.device_listbox.configure(yscrollcommand=scrollbar.set)

        self.device_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Double-click to select
        self.device_listbox.bind('<Double-Button-1>', self.on_device_select)

        # Select button
        ttk.Button(device_frame, text="Switch to Selected", command=self.on_device_select).pack(pady=(10, 0))

        # === BLACKHOLE SECTION ===
        bh_frame = ttk.LabelFrame(main_frame, text="BlackHole Setup", padding="10")
        bh_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            bh_frame,
            text="To route audio to both speakers AND BlackHole,\n"
                 "you need a Multi-Output Device.",
            justify=tk.LEFT
        ).pack(anchor=tk.W)

        btn_frame = ttk.Frame(bh_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="Open Audio MIDI Setup",
            command=create_multi_output_device
        ).pack(side=tk.LEFT)

        ttk.Button(
            btn_frame,
            text="📋 How To",
            command=self.show_howto
        ).pack(side=tk.LEFT, padx=(10, 0))

        # === BOTTOM BUTTONS ===
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X)

        ttk.Button(bottom_frame, text="🔄 Refresh", command=self.refresh_all).pack(side=tk.LEFT)
        ttk.Button(bottom_frame, text="❌ Quit", command=self.root.quit).pack(side=tk.RIGHT)

    def show_howto(self):
        """Show instructions for setting up multi-output device."""
        howto = """How to set up BlackHole Multi-Output:

1. Click 'Open Audio MIDI Setup'

2. Click the '+' button at bottom left

3. Select 'Create Multi-Output Device'

4. Check both:
   ✓ Your speakers (e.g., MacBook Pro Speakers)
   ✓ BlackHole 2ch (or 16ch)

5. (Optional) Right-click → 'Use This Device For Sound Output'
   Or select it from this app's device list

6. Name it something memorable like 'Speakers + BlackHole'

Now audio goes to both your speakers AND BlackHole!
Your DJ Lights app can capture from BlackHole."""

        messagebox.showinfo("BlackHole Setup Guide", howto)

    def update_device_list(self):
        """Update the device listbox."""
        self.device_listbox.delete(0, tk.END)

        # Try SwitchAudioSource first (most reliable)
        devices = list_switchaudio_devices()

        if not devices:
            # Fallback to system_profiler
            device_info = get_audio_devices()
            devices = [d["name"] for d in device_info]

        for device in devices:
            self.device_listbox.insert(tk.END, device)

        # Update current device
        current = get_current_output()
        self.current_device_label.config(text=current)

        # Highlight current device in list
        for i, device in enumerate(devices):
            if device == current:
                self.device_listbox.selection_set(i)
                self.device_listbox.see(i)
                break

    def update_volume_display(self):
        """Update volume slider and label."""
        vol = get_volume()
        self.volume_var.set(vol)

        muted = is_muted()
        if muted:
            self.volume_label.config(text="🔇 Muted")
            self.mute_btn.config(text="🔊 Unmute")
        else:
            self.volume_label.config(text=f"{vol}%")
            self.mute_btn.config(text="🔇 Mute")

    def on_volume_change(self, event=None):
        """Handle volume slider change."""
        vol = self.volume_var.get()
        set_volume(vol)
        self.volume_label.config(text=f"{vol}%")

    def on_mute(self):
        """Handle mute button click."""
        new_state = toggle_mute()
        self.update_volume_display()

    def on_device_select(self, event=None):
        """Handle device selection."""
        selection = self.device_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a device first")
            return

        device_name = self.device_listbox.get(selection[0])
        success = set_output_device(device_name)

        if success:
            self.current_device_label.config(text=device_name)
        else:
            # SwitchAudioSource not available, opened System Preferences
            messagebox.showinfo(
                "Manual Selection Required",
                f"Please select '{device_name}' in the Sound preferences that just opened.\n\n"
                "For automatic switching, install:\n"
                "brew install switchaudio-osx"
            )

        self.update_device_list()

    def refresh_all(self):
        """Refresh all displays."""
        self.update_device_list()
        self.update_volume_display()

    def auto_refresh(self):
        """Auto-refresh every 2 seconds."""
        self.update_volume_display()
        # Update current device label
        current = get_current_output()
        self.current_device_label.config(text=current)
        self.root.after(2000, self.auto_refresh)


def main():
    # Check if tkinter is available
    try:
        root = tk.Tk()

        # Set app icon and appearance
        root.configure(bg='#f0f0f0')

        # macOS specific styling
        try:
            from tkmacosx import Button  # Optional nicer buttons
        except ImportError:
            pass

        app = AudioRouterApp(root)
        root.mainloop()

    except tk.TclError as e:
        print(f"Error: Could not create GUI window: {e}")
        print("\nRunning in command-line mode instead...")
        cli_mode()


def cli_mode():
    """Simple command-line fallback."""
    print("\n=== Audio Router (CLI Mode) ===\n")

    # Show current state
    print(f"Current volume: {get_volume()}%")
    print(f"Muted: {is_muted()}")
    print(f"Current output: {get_current_output()}")

    print("\nAvailable devices:")
    devices = list_switchaudio_devices()
    if not devices:
        print("  (Install SwitchAudioSource for device listing)")
        print("  brew install switchaudio-osx")
    else:
        for i, d in enumerate(devices):
            print(f"  [{i}] {d}")

    print("\nCommands:")
    print("  vol N    - Set volume to N%")
    print("  mute     - Toggle mute")
    print("  switch N - Switch to device N")
    print("  q        - Quit")

    while True:
        try:
            cmd = input("\n> ").strip().lower()

            if cmd == 'q':
                break
            elif cmd.startswith('vol '):
                try:
                    vol = int(cmd.split()[1])
                    set_volume(vol)
                    print(f"Volume set to {vol}%")
                except (ValueError, IndexError):
                    print("Usage: vol N (where N is 0-100)")
            elif cmd == 'mute':
                new_state = toggle_mute()
                print(f"Muted: {new_state}")
            elif cmd.startswith('switch '):
                try:
                    idx = int(cmd.split()[1])
                    if devices and 0 <= idx < len(devices):
                        set_output_device(devices[idx])
                        print(f"Switched to: {devices[idx]}")
                    else:
                        print("Invalid device index")
                except (ValueError, IndexError):
                    print("Usage: switch N (where N is device number)")
            else:
                print("Unknown command. Try: vol, mute, switch, q")

        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break


if __name__ == "__main__":
    main()

# Setup Guide: Interview AI Assistant

This guide covers everything you need to set up to get the `interview_assistant.py` script running correctly. The script captures your microphone and your computer's system audio (so it can hear both you and the interviewer), transcribes it in real-time using either Google Cloud STT or a free local Whisper model, and generates AI coaching advice via Google Gemini.

---

## 1. System Requirements

- **Operating System:** macOS (recommended due to the BlackHole setup) or Linux/Windows with equivalent loopback audio setup.
- **Python:** Python 3.8 or higher.
- **Terminal:** Any standard terminal or command prompt (the script features an interactive UI that works well in most modern terminal emulators).

## 2. Audio Routing Setup (macOS)

To capture the interviewer's voice (the audio coming out of your speakers/headphones), the script needs a "loopback" virtual audio device. On macOS, this is commonly done using **BlackHole**.

1. **Install BlackHole:**
   - Using Homebrew (recommended):
     ```bash
     brew install blackhole-2ch
     ```
   - Or download it directly from the [Existential Audio website](https://existential.audio/blackhole/).

2. **Configure Multi-Output Device (Audio MIDI Setup):**
   To hear the interviewer while simultaneously sending their audio to BlackHole (and into the Python script), you must create a Multi-Output device:
   - Open **Audio MIDI Setup** (found in `Applications > Utilities`).
   - Click the **+** button in the bottom left corner and select **Create Multi-Output Device**.
   - Check the boxes for:
     1. Your actual headphones/speakers (e.g., "External Headphones" or "MacBook Pro Speakers").
     2. "BlackHole 2ch".
   - Make sure your headphones/speakers are set as the "Master Device" at the top of the window.
   - *Optional:* Right-click your new "Multi-Output Device" in the left sidebar and rename it (e.g., "Interview Audio Setup").

3. **Select your Audio Output:**
   - In your System Settings or by clicking the volume icon in your macOS menu bar, change your system's Audio Output to the **Multi-Output Device** you just created.
   - Now, any audio played by your computer (e.g., Google Meet, Zoom) will go to both your headphones and to BlackHole.

## 3. API Credentials & Local STT Setup

The script relies on Google Gemini for the AI coaching, and allows you to choose between Google Cloud Speech-to-Text (paid, fast) or `faster-whisper` (free, local) for transcription.

### A. (Optional) Free Local Transcription with Whisper

If you prefer to use free, local transcription instead of Google Cloud STT, you can run the script with the `--use-whisper` flag.

1. Ensure you have installed the optional `faster-whisper` dependency (included in `requirements.txt`).
2. When you run the script, use: `python interview_assistant.py --use-whisper`
3. *Note: The first time you run this, it will download the Whisper "base" model. Local transcription uses your CPU/GPU and does not require Google Cloud STT credentials.*

### B. Google Cloud Speech-to-Text (STT) - Default

If you do not use `--use-whisper`, you need to set up Google Cloud STT.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Navigate to **APIs & Services > Library**.
4. Search for **Cloud Speech-to-Text API** and click **Enable**.
5. Navigate to **APIs & Services > Credentials**.
6. Click **Create Credentials** > **Service Account**.
7. Fill in the details, click **Create and Continue**, grant the role `Cloud Speech Administrator` (or `Project > Owner` for simplicity in testing), and click **Done**.
8. Click on the newly created Service Account, go to the **Keys** tab, click **Add Key** > **Create new key**.
9. Select **JSON** and click **Create**. A `.json` file will download to your computer.
10. Move this JSON file to your project directory or a secure location. 

### C. Google Gemini AI API (Required for Coaching)

1. Go to Google AI Studio ([aistudio.google.com](https://aistudio.google.com/)).
2. Sign in with your Google account.
3. Click **Get API key** (usually in the left sidebar).
4. Click **Create API key** > select your Google Cloud project (or create a new one).
5. Copy the generated API Key.

## 4. Environment Variables (`.env`)

You need to tell the Python script where to find your credentials. Create a file named `.env` in the same directory as `interview_assistant.py` and populate it with your keys.

1. Create a `.env` file:
   ```bash
   touch .env
   ```
2. Open it in a text editor and add the following lines:
   ```env
   # Path to your downloaded Google Cloud Service Account JSON key
   GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-file.json"

   # Your Gemini API Key from Google AI Studio
   GEMINI_API_KEY="your_gemini_api_key_here"
   ```

## 5. Install Python Dependencies

Make sure you are in the project folder and run:

```bash
pip install -r requirements.txt
```

*Note: If `requirements.txt` is missing, the necessary packages are:*
```bash
pip install soundcard numpy google-cloud-speech google-genai python-dotenv faster-whisper
```

## 6. Optional: Transparent Terminal with Ghostty (macOS)

If you want to use the AI script in a transparent terminal so it can float unobtrusively over your video call, you can use the Ghostty terminal emulator.

1. **Install Ghostty:**
   ```bash
   brew install --cask ghostty@tip
   ```
   *(Source: https://formulae.brew.sh/cask/ghostty@tip)*

2. **Configure Ghostty:**
   Ghostty is configured via a text-based configuration file. Create or edit the config file located at `~/Library/Application Support/com.mitchellh.ghostty/config.ghostty` (create the directory if it doesn't exist).

   Add the following configuration for transparency and optimized interview viewing:

   ```ini
   # Transparency settings
   background-opacity = 0.0
   background-blur-radius = 20

   # Window UI settings
   window-decoration = false
   confirm-close-surface = false
   window-padding-balance = true

   # Core Font Settings
   font-family = "JetBrainsMono Nerd Font Mono"
   font-size = 16
   
   # Fallback Fonts (repeatable for missing glyphs/symbols)
   # font-family = "Symbols Nerd Font Mono"
   # font-family = "Noto Sans CJK JP"

   # Font features (ligatures) and styling
   # font-feature = calt
   # font-feature = liga
   
   # Fine-Tuning Spacing
   # adjust-cell-height = 2
   
   # View available themes with: ghostty +list-themes
   # theme = <theme-name>
   
   # Color mappings (Optional: customize specific ANSI colors if you want to change the AI text color, which uses ANSI green by default)
   # palette = 2=#00FF00
   
   # Custom keybindings (example)
   # key_map = ctrl+a>n new_tab
   ```

3. **Usage Tips:**
   - Reload your config changes instantly with `Cmd+Shift+,`
   - Adjust opacity on the fly inside the python script using `<` to decrease opacity, and `>` to increase opacity. This natively modifies the `~/Library/Application Support/com.mitchellh.ghostty/config.ghostty` file and triggers an AppleScript to "Reload Configuration" directly in Ghostty.

## 7. Running the Assistant

Once everything is set up:
1. Ensure your system Audio Output is set to the Multi-Output Device (BlackHole + Headphones).
2. Run the script:
   - For Google Cloud STT:
     ```bash
     python interview_assistant.py
     ```
   - For free local STT:
     ```bash
     python interview_assistant.py --use-whisper
     ```
3. The terminal will clear and show a live stream. You will see `🎙 ME` for your voice, `🔊 THEM` for the system audio (BlackHole), and `🤖 AI` for the Gemini coaching tips!
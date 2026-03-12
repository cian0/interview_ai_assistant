import soundcard as sc
import numpy as np
import threading
import queue
from google.cloud import speech
from google import genai
from google.genai import types
import time
import sys
import os
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
import shutil
import select
import tty
import termios
import signal
import argparse
import platform
from dotenv import load_dotenv
from PIL import ImageGrab

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


# --- Config ---
SAMPLE_RATE = 16000
CHUNK_FRAMES = int(SAMPLE_RATE * 0.1)

AVAILABLE_MODELS = ["gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview"]
GEMINI_MODEL = AVAILABLE_MODELS[0]

# Whisper Config
WHISPER_CHUNK_SEC = 5
WHISPER_OVERLAP_SEC = 1
WHISPER_STEP_FRAMES = SAMPLE_RATE * (WHISPER_CHUNK_SEC - WHISPER_OVERLAP_SEC)
WHISPER_FULL_BUFFER_FRAMES = SAMPLE_RATE * WHISPER_CHUNK_SEC
WHISPER_MODEL_SIZE = "base"
# GEMINI_MODEL = "gemini-3.1-pro-preview"  # Latest Gemini 2.5 Pro

GEMINI_PAUSE_THRESHOLD = 2.5  # seconds of silence before AI comments
CONTEXT_MODE = "SINCE_LAST_AI"  # "SINCE_LAST_AI" or "EXCHANGES"

# --- Queues ---
mic_q = queue.Queue()
sys_q = queue.Queue()

# --- Printing ---
print_lock = threading.Lock()
console = Console()

transcript_history = []
active_interims = {}
last_update_time = time.time()
is_paused = False
mic_muted = True
gemini_triggered = False  # prevent double-firing per pause
current_gemini_thread_id = 0  # To track and cancel superseded streams
load_dotenv()
gemini_client = genai.Client()

captured_screenshots = []
active_chat_session = None

USE_WHISPER = False
whisper_model = None
whisper_model_lock = threading.Lock()

def init_whisper():
    global whisper_model
    if not USE_WHISPER:
        return
    if WhisperModel is None:
        print("faster-whisper is not installed. Run `pip install faster-whisper` or run without --use-whisper.")
        sys.exit(1)
    print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print("Model loaded.")

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95
    return audio.astype(np.float32)


def redraw_console():
    term_width, term_height = shutil.get_terminal_size((80, 24))
    num_interims = len(active_interims)
    
    # We will print everything using rich console to handle markdown
    sys.stdout.write('\033[2J\033[H')
    console.print("Streaming to Google STT + Gemini AI... Press Ctrl+C to stop.")
    console.print("-" * min(term_width, 80))

    # To avoid taking up too much vertical space, we only show recent history
    # But because markdown might span multiple lines, we just take the last few entries
    # instead of exact line count. Let's take the last 5 entries.
    for line in transcript_history[-5:]:
        if "[🤖 AI]" in line:
            # Extract the actual text by removing the label part
            # line looks like "[🤖 AI] ✅ Some text" or just "[🤖 AI] Some text"
            label = "[🤖 AI] ✅" if "✅" in line else "[🤖 AI]"
            text_part = line.replace(label, "").strip()

            console.print(f"[bold green]{label}[/bold green]")
            console.print(Markdown(text_part, style="green"))
            console.print("") # spacing
        elif "[🗣️ YOU ASKED]" in line:
            console.print(f"[bold cyan]{line}[/bold cyan]")
            console.print("") # spacing
        else:
            console.print(line)

    if active_interims:
        console.print("\n")
        for l, t in active_interims.items():
            if l == "🤖 AI":
                console.print(f"[bold green][{l}] 💬[/bold green]")
                console.print(Markdown(t, style="green"))
            else:
                console.print(f"[{l}] 💬 {t}")

    is_ghostty = os.environ.get("TERM_PROGRAM") == "ghostty" or os.environ.get("TERM") == "xterm-ghostty"

    screenshot_str = f" | {len(captured_screenshots)} 📸" if captured_screenshots else ""
    chat_hist_str = ""

    if is_ghostty:
        controls_str = f"m: unmute | r: retry AI | a: ask AI | s: switch model | p: snap screen | c: clear screens | \"<\" or \">\": opacity{screenshot_str}{chat_hist_str}" if mic_muted else f"m: mute | r: retry AI | a: ask AI | s: switch model | p: snap screen | c: clear screens | </>{screenshot_str}{chat_hist_str}"
    else:
        controls_str = f"m: unmute | r: retry AI | a: ask AI | s: switch model | p: snap screen | c: clear screens{screenshot_str}{chat_hist_str}" if mic_muted else f"m: mute | r: retry AI | a: ask AI | s: switch model | p: snap screen | c: clear screens{screenshot_str}{chat_hist_str}"

    model_name = "Pro" if "pro" in GEMINI_MODEL else "Flash"

    if mic_muted:
        console.print(f"\n[STATUS] 🔇 MIC MUTED ({controls_str}) | 🧠 {model_name}")
    elif is_paused:
        console.print(f"\n[STATUS] ⏸️  PAUSE... ({controls_str.replace('unmute', 'mute')}) | 🧠 {model_name}")
    else:
        console.print(f"\n[STATUS] 🎙️ LISTENING ({controls_str}) | 🧠 {model_name}")

in_settings_menu = False
in_ask_mode_input = False

def draw_settings_menu():
    term_width, term_height = shutil.get_terminal_size((80, 24))
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.write("--- GHOSTTY SETTINGS MENU ---\n")
    sys.stdout.write("1. Font Size (e.g., 14)\n")
    sys.stdout.write("2. AI Text Color (e.g., #00FF00)\n")
    sys.stdout.write("3. Font Family (e.g., JetBrains Mono)\n")
    sys.stdout.write("4. Adjust Cell Height (e.g., 2, 10%)\n")
    sys.stdout.write("b. Back to Interview\n")
    sys.stdout.write("\nSelect an option: ")
    sys.stdout.flush()

def handle_settings_input(option):
    global in_settings_menu
    
    if option.lower() == 'b':
        in_settings_menu = False
        redraw_console()
        return

    mapping = {
        '1': ('font-size', 'Enter new font-size (e.g. 14, 16): '),
        '2': ('palette = 2', 'Enter new HEX color for AI (e.g. #00FF00, #FF00FF): '),
        '3': ('font-family', 'Enter new font-family (e.g. "JetBrains Mono"): '),
        '4': ('adjust-cell-height', 'Enter new cell height (e.g. 2, 10%): '),
    }
    
    if option in mapping:
        key, prompt = mapping[option]
        sys.stdout.write(f"\n{prompt}")
        sys.stdout.flush()
        
        # Disable cbreak temporarily to allow normal input
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            val = input().strip()
            if val:
                update_ghostty_config_key(key, val)
        finally:
            tty.setcbreak(fd)
            
        draw_settings_menu()
    else:
        draw_settings_menu()

def update_ghostty_config_key(key, value):
    # macOS typical path. For Linux, Ghostty usually uses ~/.config/ghostty/config
    is_mac = platform.system() == "Darwin"
    if is_mac:
        config_dir = os.path.expanduser("~/Library/Application Support/com.mitchellh.ghostty")
    else:
        config_dir = os.path.expanduser("~/.config/ghostty")
        
    config_path = os.path.join(config_dir, "config" if not is_mac else "config.ghostty")
    
    try:
        import re
        import subprocess
        
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        content = ""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                content = f.read()
                
        # Handle the special palette key
        if key == 'palette = 2':
            pattern = r'^palette\s*=\s*2\s*=.*$'
            new_line = f"palette = 2={value}"
        else:
            pattern = rf'^{re.escape(key)}\s*=.*$'
            new_line = f"{key} = {value}"
            
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            content = re.sub(pattern, new_line, content, flags=re.MULTILINE)
        else:
            if content and not content.endswith('\n'):
                content += '\n'
            content += f"{new_line}\n"
            
        with open(config_path, 'w') as f:
            f.write(content)
            
        if is_mac:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.scpt', delete=False) as temp_script:
                temp_script.write('tell application "System Events" to tell process "Ghostty" to click menu item "Reload Configuration" of menu "Ghostty" of menu bar item "Ghostty" of menu bar 1')
                temp_script_path = temp_script.name
                
            os.system(f"osascript {temp_script_path} >/dev/null 2>&1 && rm {temp_script_path} &")
        else:
            # On Linux, Ghostty auto-reloads or requires manual reload. We just write the config.
            pass
    except Exception as e:
        print(f"Error updating config: {e}")
        pass


def thread_safe_print(label, text, is_final=False):
    global last_update_time, is_paused, gemini_triggered, current_gemini_thread_id
    with print_lock:
        last_update_time = time.time()
        is_paused = False
        gemini_triggered = False  # reset so AI can comment on the next pause
        current_gemini_thread_id += 1  # cancel any running AI stream

        if is_final:
            transcript_history.append(f"[{label}] ✅ {text}")
            if label in active_interims:
                del active_interims[label]
        else:
            active_interims[label] = text

        if not in_ask_mode_input:
            redraw_console()


def build_conversation_context():
    """Build a readable transcript from history for Gemini."""
    if not transcript_history:
        return None
        
    recent = []
    them_block_count = 0
    last_speaker = None
    
    # Iterate backwards (newest to oldest)
    for line in reversed(transcript_history):
        if "[🤖 AI]" in line:
            if CONTEXT_MODE == "SINCE_LAST_AI" and recent:
                # Hit a previous AI response after collecting some lines. Cut context here.
                break
            # Otherwise (EXCHANGES mode, or we are retrying and haven't collected anything yet), ignore it
            continue
            
        # Filter out lines that are empty after the checkmark
        if "✅" not in line or not line.split("✅")[-1].strip():
            continue
            
        if "[🔊 THEM]" in line:
            speaker = "THEM"
        elif "[🎙 ME]" in line:
            speaker = "ME"
        else:
            continue
            
        if CONTEXT_MODE == "EXCHANGES":
            if speaker != last_speaker:
                if speaker == "THEM":
                    them_block_count += 1
                last_speaker = speaker
                
            # Stop if we have collected 3 THEM blocks and are about to start a ME block
            if them_block_count == 3 and speaker == "ME":
                break
            # Or stop if we somehow exceed 3 THEM blocks
            if them_block_count > 3:
                break
                
        recent.append(line)
        
        if CONTEXT_MODE == "EXCHANGES" and len(recent) >= 15:
            break
            
    if not recent:
        return None
            
    # Return the context, which is already ordered newest at the top
    return "\n".join(recent)


def gemini_comment_stream(thread_id, ask_question=None):
    """Runs in its own thread. Streams a Gemini comment as a fake STT speaker."""
    global gemini_triggered, captured_screenshots, active_chat_session

    context = build_conversation_context()

    if not context and not ask_question:
        return

    label = "🤖 AI"
    system_prompt = (
        "You are a smart, concise real-time technical interview coach listening in on an interview. "
        "The conversation involves 'ME' (the interviewee) and 'THEM' (the interviewer). "
        "Your job is to provide coaching and proper answers to the questions being asked by the interviewer. "
        "The conversation history is provided from NEWEST to OLDEST. "
        "PRIORITIZE the very first line (the most recent input) to focus your answer on, and use the older lines "
        "only as supporting context if applicable. If there are multiple unrelated questions asked, focus only on the first one. "
        "Keep your advice concise, SIMPLIFY your suggested answers using bullet points. Let the reader expand on the simple bullet points instead of explaining it thoroughly. Each bullet point should have a maximum of 10 words only, hence use keywords and queues that can easily be picked up"
        "Be direct and helpful. IMPORTANT: Always write complete sentences and never trail off or end mid-thought."
    )

    if ask_question:
        # Override the system prompt during Ask mode to be more helpful and detailed
        # instead of the restrictive bullet-point coach mode.
        system_prompt = (
            "You are an expert programmer and technical interview coach. "
            "When given a problem, question, or screenshot, explain the logic and your reasoning clearly. "
            "Provide complete, working code solutions when applicable with comments. "
            "Use markdown formatting with headers and code blocks."
        )

        full_context = "\n".join(transcript_history)
        context_text = f"Full Session Transcript:\n{full_context}\n\n" if full_context else "No conversation history yet.\n\n"

        if captured_screenshots:
            user_prompt = f"{context_text}The user directly asked you this question: '{ask_question}'. Please answer their question directly, using the conversation context and the {len(captured_screenshots)} provided screenshots if needed."
            with print_lock:
                transcript_history.append(f"[🗣️ YOU ASKED] {ask_question} ({len(captured_screenshots)} screenshots)")
                redraw_console()

            contents = []
            for img in captured_screenshots:
                contents.append(img)
            contents.append(user_prompt)
            captured_screenshots.clear()
        else:
            user_prompt = f"{context_text}The user directly asked you this question: '{ask_question}'. Please answer their question directly, using the conversation context if needed."
            with print_lock:
                transcript_history.append(f"[🗣️ YOU ASKED] {ask_question}")
                redraw_console()

            contents = user_prompt
    else:
        user_prompt = f"Conversation (newest to oldest):\n{context}\n\nGive your coaching advice and suggested answer now."
        contents = user_prompt

    try:
        accumulated = ""

        if ask_question:
            # Use Ask mode config
            generate_config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=8192*2 if "pro" in GEMINI_MODEL.lower() else 800,
                thinking_config=types.ThinkingConfig(thinking_budget=-1) if "pro" in GEMINI_MODEL.lower() else None
            )

            # We use stateless stream with the full history as context injected manually
            stream = gemini_client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config=generate_config
            )
        else:
            # Prepare config based on model type
            config_kwargs = {
                "system_instruction": system_prompt,
                "max_output_tokens": 800,
            }

            # If it's a Pro model, configure the thinking parameters to make it concise
            if "pro" in GEMINI_MODEL.lower():
                # During normal coaching mode, we enforce concise answers
                config_kwargs["system_instruction"] = system_prompt + "\nBe extremely concise. No fluff. Give clear, brief answers."
                config_kwargs["max_output_tokens"] = 2000 # keep it short
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=8192)

            generate_config = types.GenerateContentConfig(**config_kwargs)
            
            # Normal stateless coaching stream
            # DO NOT use chats.create() here, use generate_content_stream so it remains stateless
            stream = gemini_client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config=generate_config,
            )

        for chunk in stream:
            if current_gemini_thread_id != thread_id:
                # User started speaking, cancel this stream
                break
                
            if chunk.text:
                accumulated += chunk.text
                # Stream each chunk as an interim update
                with print_lock:
                    if current_gemini_thread_id == thread_id:
                        active_interims[label] = accumulated
                        redraw_console()

        # Finalize as a completed transcript line
        with print_lock:
            if current_gemini_thread_id == thread_id:
                if accumulated.strip():
                    transcript_history.append(f"[{label}] ✅ {accumulated.strip()}")
                if label in active_interims:
                    del active_interims[label]
                redraw_console()
            else:
                # Stream was interrupted, just clean up interim if we own it
                if label in active_interims and active_interims[label] == accumulated:
                    del active_interims[label]
                    redraw_console()

    except Exception as e:
        with print_lock:
            transcript_history.append(f"[{label}] ⚠️ Gemini error: {e}")
            if label in active_interims:
                del active_interims[label]
            redraw_console()


def pause_monitor():
    global is_paused, gemini_triggered
    while True:
        time.sleep(0.5)
        with print_lock:
            elapsed = time.time() - last_update_time
            if not is_paused and elapsed > 0.5:
                is_paused = True
                if not in_ask_mode_input:
                    redraw_console()

            # Trigger Gemini once per pause, after GEMINI_PAUSE_THRESHOLD seconds
            if is_paused and not gemini_triggered and elapsed > GEMINI_PAUSE_THRESHOLD:
                gemini_triggered = True  # lock to prevent re-triggering
                if not in_ask_mode_input:
                    # Launch in a separate thread so pause_monitor stays non-blocking
                    thread_id = current_gemini_thread_id
                    t = threading.Thread(target=gemini_comment_stream, args=(thread_id,), daemon=True)
                    t.start()


def capture_mic():
    try:
        mic = sc.default_microphone()
        print(f"[MIC] Capturing from: {mic.name}")
        
        if USE_WHISPER:
            buffer = np.zeros(WHISPER_FULL_BUFFER_FRAMES, dtype=np.float32)
        
        with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
            while True:
                if USE_WHISPER:
                    new_data = rec.record(numframes=WHISPER_STEP_FRAMES).flatten()
                    if mic_muted:
                        new_data = np.zeros_like(new_data)
                    buffer[:SAMPLE_RATE * WHISPER_OVERLAP_SEC] = buffer[WHISPER_STEP_FRAMES:]
                    buffer[SAMPLE_RATE * WHISPER_OVERLAP_SEC:] = new_data
                    mic_q.put(buffer.copy())
                else:
                    data = rec.record(numframes=CHUNK_FRAMES).flatten()
                    if mic_muted:
                        data = np.zeros_like(data)
                    mic_q.put(data)
    except Exception as e:
        print(f"[MIC] Error: {e}")


def capture_system(sys_audio_device=None):
    try:
        try:
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        except Exception:
            loopback = None

        if loopback is None:
            all_mics = sc.all_microphones(include_loopback=True)
            if sys_audio_device:
                loopback = next((m for m in all_mics if sys_audio_device.lower() in m.name.lower()), None)
            
            if loopback is None:
                # Fallback OS-specific defaults
                if platform.system() == "Darwin":
                    loopback = next((m for m in all_mics if "BlackHole" in m.name), None)
                elif platform.system() == "Linux":
                    # Look for PulseAudio/PipeWire monitor devices
                    loopback = next((m for m in all_mics if "Monitor" in m.name or "Loopback" in m.name), None)

        if loopback is None:
            print("[SYS] Error: System audio loopback device not found.")
            return

        print(f"[SYS] Capturing from: {loopback.name}")
        
        if USE_WHISPER:
            buffer = np.zeros(WHISPER_FULL_BUFFER_FRAMES, dtype=np.float32)
            
        with loopback.recorder(samplerate=SAMPLE_RATE, channels=2) as rec:
            while True:
                if USE_WHISPER:
                    new_data = rec.record(numframes=WHISPER_STEP_FRAMES)
                    new_data_mono = new_data.mean(axis=1)
                    buffer[:SAMPLE_RATE * WHISPER_OVERLAP_SEC] = buffer[WHISPER_STEP_FRAMES:]
                    buffer[SAMPLE_RATE * WHISPER_OVERLAP_SEC:] = new_data_mono
                    sys_q.put(buffer.copy())
                else:
                    data = rec.record(numframes=CHUNK_FRAMES).mean(axis=1)
                    sys_q.put(data)
    except StopIteration:
        print("[SYS] Error: BlackHole device not found.")
    except Exception as e:
        print(f"[SYS] Error: {e}")


def audio_generator(audio_queue):
    while True:
        chunk = audio_queue.get()
        audio_int16 = (chunk * 32767).astype(np.int16)
        yield audio_int16.tobytes()


def transcribe_whisper(audio_queue, label):
    print(f"[{label}] Transcribe thread started (Whisper).")
    while True:
        try:
            audio_data = audio_queue.get()
            
            # Apply Normalization
            audio = normalize_audio(audio_data)
            
            if np.abs(audio).mean() < 0.001:  
                continue
                
            with whisper_model_lock:
                segments, _ = whisper_model.transcribe(
                    audio,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=400,
                        threshold=0.5,
                    ),
                    temperature=[0.0, 0.2, 0.4],
                    condition_on_previous_text=True,
                )
            for seg in segments:
                text = seg.text.strip()
                if text:
                    thread_safe_print(label, text, is_final=True)
        except Exception as e:
            with print_lock:
                print(f"[{label}] Transcription error: {e}")


def transcribe_stream(audio_queue, label):
    print(f"[{label}] Connecting to Google STT...")
    try:
        client = speech.SpeechClient()
    except Exception as e:
        print(f"[{label}] Error initializing client: {e}")
        return

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        model="latest_long",
        enable_automatic_punctuation=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    while True:
        try:
            requests = (
                speech.StreamingRecognizeRequest(audio_content=chunk)
                for chunk in audio_generator(audio_queue)
            )
            responses = client.streaming_recognize(streaming_config, requests)
            last_interim = {}

            for response in responses:
                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript.strip()
                    if result.is_final:
                        thread_safe_print(label, transcript, is_final=True)
                        last_interim[label] = ""
                    else:
                        if last_interim.get(label) != transcript:
                            thread_safe_print(label, transcript, is_final=False)
                            last_interim[label] = transcript

        except Exception as e:
            with print_lock:
                print(f"\n[{label}] Stream error (reconnecting): {e}")
            time.sleep(1)


def key_listener():
    global mic_muted, current_gemini_thread_id, gemini_triggered, in_ask_mode_input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                char = sys.stdin.read(1)
                is_ghostty = os.environ.get("TERM_PROGRAM") == "ghostty" or os.environ.get("TERM") == "xterm-ghostty"

                # if in_settings_menu:
                #     handle_settings_input(char)
                # elif char.lower() == 's' and is_ghostty:
                #     in_settings_menu = True
                #     draw_settings_menu()
                # el
                if char.lower() == 'm':
                    mic_muted = not mic_muted
                    with print_lock:
                        redraw_console()
                elif char.lower() == 'r':
                    with print_lock:
                        current_gemini_thread_id += 1
                        thread_id = current_gemini_thread_id
                        gemini_triggered = True
                    t = threading.Thread(target=gemini_comment_stream, args=(thread_id,), daemon=True)
                    t.start()
                elif char.lower() == 's':
                    global GEMINI_MODEL
                    current_idx = AVAILABLE_MODELS.index(GEMINI_MODEL)
                    next_idx = (current_idx + 1) % len(AVAILABLE_MODELS)
                    GEMINI_MODEL = AVAILABLE_MODELS[next_idx]
                    with print_lock:
                        transcript_history.append(f"[STATUS] Switched model to {GEMINI_MODEL}")
                        redraw_console()
                elif char.lower() == 'p':
                    # Capture a screenshot silently
                    screenshot = ImageGrab.grab()
                    screenshot.thumbnail((1024, 1024))
                    captured_screenshots.append(screenshot)
                    with print_lock:
                        redraw_console()
                elif char.lower() == 'c':
                    # Clear all captured screenshots AND the active chat session
                    captured_screenshots.clear()
                    with print_lock:
                        transcript_history.clear()
                        active_interims.clear()
                        transcript_history.append("[STATUS] 🗑️ Cleared all context, history, and screenshots.")
                        redraw_console()
                elif char.lower() == 'a':
                    # Ask mode: disable cbreak, loop asking for input until 'x' or 'exit'
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        while True:
                            in_ask_mode_input = True
                            sys.stdout.write("\n\033[93m[ASK AI]\033[0m Enter your question (or 'x'/'exit' to quit): ")
                            sys.stdout.flush()
                            question = input().strip()
                            in_ask_mode_input = False

                            if question.lower() in ['x', 'exit', 'quit']:
                                with print_lock:
                                    transcript_history.append("[STATUS] 🚪 Exited Ask Mode.")
                                    redraw_console()
                                break

                            if question or captured_screenshots:
                                with print_lock:
                                    current_gemini_thread_id += 1
                                    thread_id = current_gemini_thread_id
                                    gemini_triggered = True

                                # We run the Gemini request and wait for it to finish, so the response
                                # streams into the UI cleanly before we prompt for the next question.
                                t = threading.Thread(target=gemini_comment_stream, args=(thread_id, question), daemon=True)
                                t.start()
                                t.join()
                            else:
                                with print_lock:
                                    redraw_console()
                    finally:
                        in_ask_mode_input = False
                        tty.setcbreak(fd)
                elif (char == '<' or char == '>') and is_ghostty:
                    # Instead of using subprocess which forks and crashes gRPC, 
                    # create a background thread to update the config and reload Ghostty
                    def update_ghostty_opacity(char_pressed):
                        is_mac = platform.system() == "Darwin"
                        if is_mac:
                            config_dir = os.path.expanduser("~/Library/Application Support/com.mitchellh.ghostty")
                            config_path = os.path.join(config_dir, "config.ghostty")
                        else:
                            config_dir = os.path.expanduser("~/.config/ghostty")
                            config_path = os.path.join(config_dir, "config")
                            
                        try:
                            import re
                            import subprocess
                            
                            if not os.path.exists(config_dir):
                                os.makedirs(config_dir)
                                
                            content = ""
                            if os.path.exists(config_path):
                                with open(config_path, 'r') as f:
                                    content = f.read()
                            
                            current_opacity = 0.0
                            match = re.search(r'^background-opacity\s*=\s*([0-9.]+)', content, re.MULTILINE)
                            if match:
                                current_opacity = float(match.group(1))
                            
                            if char_pressed == '<':
                                new_opacity = max(0.0, current_opacity - 0.1)
                            else:
                                new_opacity = min(1.0, current_opacity + 0.1)
                                
                            new_opacity_str = f"{new_opacity:.1f}"
                            
                            if match:
                                content = re.sub(r'^background-opacity\s*=\s*[0-9.]+', f'background-opacity = {new_opacity_str}', content, flags=re.MULTILINE)
                            else:
                                if content and not content.endswith('\n'):
                                    content += '\n'
                                content += f'background-opacity = {new_opacity_str}\n'
                                
                            with open(config_path, 'w') as f:
                                f.write(content)
                                
                            if is_mac:
                                # Write the AppleScript to a temporary file, and use os.system in the background using '&'
                                # to avoid any fork() issues caused by Python's subprocess module interacting with gRPC.
                                import tempfile
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.scpt', delete=False) as temp_script:
                                    temp_script.write('tell application "System Events" to tell process "Ghostty" to click menu item "Reload Configuration" of menu "Ghostty" of menu bar item "Ghostty" of menu bar 1')
                                    temp_script_path = temp_script.name
                                    
                                os.system(f"osascript {temp_script_path} >/dev/null 2>&1 && rm {temp_script_path} &")
                        except Exception:
                            pass
                            
                    update_ghostty_opacity(char)
                elif char == '\x03': # Ctrl+C
                    os.kill(os.getpid(), signal.SIGINT)
                    break
    except Exception as e:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Interview Assistant")
    parser.add_argument("--use-whisper", action="store_true", help="Use faster-whisper instead of Google STT")
    parser.add_argument("--sys-audio-device", type=str, default=None, help="Name of the system audio loopback device (e.g., 'BlackHole', 'Monitor', etc.)")
    args = parser.parse_args()

    USE_WHISPER = args.use_whisper
    
    if USE_WHISPER:
        init_whisper()
        transcribe_target = transcribe_whisper
    else:
        transcribe_target = transcribe_stream

    threads = [
        threading.Thread(target=capture_mic, daemon=True),
        threading.Thread(target=capture_system, args=(args.sys_audio_device,), daemon=True),
        threading.Thread(target=transcribe_target, args=(mic_q, "🎙 ME"), daemon=True),
        threading.Thread(target=transcribe_target, args=(sys_q, "🔊 THEM"), daemon=True),
        threading.Thread(target=pause_monitor, daemon=True),
        threading.Thread(target=key_listener, daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")

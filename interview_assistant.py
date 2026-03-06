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
import shutil
import select
import tty
import termios
import signal
import argparse
from dotenv import load_dotenv

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


# --- Config ---
SAMPLE_RATE = 16000
CHUNK_FRAMES = int(SAMPLE_RATE * 0.1)
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"  # Latest Gemini 2.5 Pro

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

transcript_history = []
active_interims = {}
last_update_time = time.time()
is_paused = False
mic_muted = True
gemini_triggered = False  # prevent double-firing per pause
current_gemini_thread_id = 0  # To track and cancel superseded streams
load_dotenv()
gemini_client = genai.Client()

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
    max_history_lines = max(5, term_height - num_interims - 5)

    sys.stdout.write('\033[2J\033[H')
    sys.stdout.write("Streaming to Google STT + Gemini AI... Press Ctrl+C to stop.\n")
    sys.stdout.write("-" * min(term_width, 80) + "\n")

    for line in transcript_history[-max_history_lines:]:
        if "[🤖 AI]" in line:
            sys.stdout.write(f"\033[92m{line}\033[0m\n")
        else:
            sys.stdout.write(f"{line}\n")

    if active_interims:
        sys.stdout.write("\n")
        for l, t in active_interims.items():
            if l == "🤖 AI":
                sys.stdout.write(f"\033[92m[{l}] 💬 {t}\033[0m\n")
            else:
                sys.stdout.write(f"[{l}] 💬 {t}\n")

    if mic_muted:
        sys.stdout.write("\n[STATUS] 🔇 MIC MUTED (m: unmute | r: retry AI)\n")
    elif is_paused:
        sys.stdout.write("\n[STATUS] ⏸️  PAUSE... (m: mute | r: retry AI)\n")
    else:
        sys.stdout.write("\n[STATUS] 🎙️ LISTENING (m: mute | r: retry AI)\n")

    sys.stdout.flush()


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


def gemini_comment_stream(thread_id):
    """Runs in its own thread. Streams a Gemini comment as a fake STT speaker."""
    global gemini_triggered

    context = build_conversation_context()
        
    if not context:
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
    user_prompt = f"Conversation (newest to oldest):\n{context}\n\nGive your coaching advice and suggested answer now."

    try:
        accumulated = ""
        for chunk in gemini_client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=800,
            ),
        ):
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
                redraw_console()

            # Trigger Gemini once per pause, after GEMINI_PAUSE_THRESHOLD seconds
            if is_paused and not gemini_triggered and elapsed > GEMINI_PAUSE_THRESHOLD:
                gemini_triggered = True  # lock to prevent re-triggering
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


def capture_system():
    try:
        try:
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        except Exception:
            loopback = None

        if loopback is None:
            loopback = next(m for m in sc.all_microphones(include_loopback=True)
                            if "BlackHole" in m.name)

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
    global mic_muted, current_gemini_thread_id, gemini_triggered
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                char = sys.stdin.read(1)
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
    args = parser.parse_args()

    USE_WHISPER = args.use_whisper
    
    if USE_WHISPER:
        init_whisper()
        transcribe_target = transcribe_whisper
    else:
        transcribe_target = transcribe_stream

    threads = [
        threading.Thread(target=capture_mic, daemon=True),
        threading.Thread(target=capture_system, daemon=True),
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

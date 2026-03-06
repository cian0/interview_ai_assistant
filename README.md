# Interview AI Assistant

Assists you with interviews by acting as a real-time, smart technical interview coach!

## What it does

This Python script (`interview_assistant.py`) streams audio from both your microphone (you) and your system audio (the interviewer) to provide real-time transcriptions and AI-generated coaching advice. 

### Key Features:
- **Dual-channel transcription:** Captures and transcribes both your microphone ("🎙 ME") and system audio ("🔊 THEM") simultaneously using either Google Cloud Speech-to-Text (default) or faster-whisper (local, free STT).
- **Real-time AI Coaching:** Automatically detects pauses in the conversation (silence threshold of 2.5 seconds) and sends the recent transcript history to Google Gemini (using the `gemini-3.1-flash-lite-preview` model by default). Gemini then acts as a technical interview coach, giving concise, bulleted advice and suggested answers based on the interviewer's most recent questions.
- **Interactive Console UI:** Provides a live, auto-updating terminal interface that differentiates between speakers and highlights AI coaching advice.
- **Manual Controls:** During the stream, you can use hotkeys directly in the terminal:
  - `m`: Mute / Unmute your microphone.
  - `r`: Force Gemini to retry generating advice based on the current context.

## APIs Used & Cost Warning

⚠️ **IMPORTANT: This script uses paid cloud APIs that may charge your account.** ⚠️

- **Google Cloud Speech-to-Text (STT):** Used for real-time dual-channel audio transcription by default. STT is billed based on the amount of audio processed. **Streaming audio continuously can consume significant API quotas and incur charges.** Please monitor your usage in the Google Cloud Console.
  - **Alternative:** You can use `--use-whisper` for free, local transcription using `faster-whisper`.
- **Google Gemini API (`gemini-3.1-flash-lite-preview`):** Used to generate real-time coaching advice. While there is a free tier for Gemini API usage (depending on your region and account status), exceeding limits or using different models (like `pro`) will incur costs.

## Prerequisites & Setup

> **[See the comprehensive SETUP.md guide here](./SETUP.md)**

For a complete walkthrough on how to install BlackHole (loopback audio), configure Google Cloud STT, get a Gemini API key, and set up your `.env` file, please see the **[Setup Guide](SETUP.md)**.

## Quick Start

1. Follow the **[Setup Guide](SETUP.md)** to configure your audio routing and API credentials.
2. Ensure your `.env` file is populated with your Google credentials/API keys.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the script:
   - For Google Cloud STT: `python interview_assistant.py`
   - For free local STT: `python interview_assistant.py --use-whisper`
5. The terminal will start displaying the real-time transcript and AI coaching notes. Press `Ctrl+C` to stop.

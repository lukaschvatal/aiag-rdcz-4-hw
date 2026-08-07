# Assignment 01: Gemini tool calling

This example implements a walking conditions assistant. It helps the user decide
whether the current weather and air quality are suitable for going for a walk.
Gemini decides whether it can answer directly or needs current data from one or
both available tools:

- `get_current_weather` returns current weather conditions,
- `get_current_air_quality` returns the European AQI and pollutant levels.

The Python application calls the public
[Open-Meteo APIs](https://open-meteo.com/en/docs), returns tool results to
Gemini, and prints Gemini's final user-friendly answer. The agent loop makes at
most five model calls. Tool use is disabled on the fifth call so that the last
call produces a final answer instead of starting more work.

The example uses the stable `gemini-3.5-flash-lite` model.

Open-Meteo is free for non-commercial use and does not require registration or
an API key. Gemini requires a `GEMINI_API_KEY`.

## Setup

Create the local environment file and add your Gemini API key:

```bash
cp .env.example .env
```

Install the dependencies and start the assistant:

```bash
uv sync
uv run main.py
```

The assistant then asks you to enter a question in the terminal:

```text
Hi! I help assess whether conditions are suitable for a walk.
Your question: Je teď v Brně vhodné počasí na procházku?
```

There is no built-in default question. An empty question ends the program.

For a question that needs current data, the output shows all three important
stages of function calling:

1. the model, tool mode, and conversation history sent in each request,
2. the response type, finish reason, and token usage,
3. the function names, call IDs, and arguments selected by Gemini,
4. the data returned by Open-Meteo and added back to the conversation,
5. the final natural-language answer produced by Gemini from that data.

The assistant starts its final response with a clear recommendation and then
explains the relevant measured conditions and practical precautions. If live
data is unnecessary, Gemini can answer directly without calling either tool.

Air-quality data is provided by Open-Meteo using CAMS ENSEMBLE data from the
Copernicus Atmosphere Monitoring Service.

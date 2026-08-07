import json
import os
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types


MODEL = "gemini-3.5-flash-lite"
MAX_MODEL_CALLS = 5
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
REQUEST_TIMEOUT_SECONDS = 10

WEATHER_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def find_location(city: str) -> dict[str, Any] | None:
    location_response = requests.get(
        GEOCODING_API_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    location_response.raise_for_status()

    locations = location_response.json().get("results", [])
    if not locations:
        return None

    return locations[0]


def format_location(location: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": location["name"],
        "country": location.get("country"),
        "latitude": location["latitude"],
        "longitude": location["longitude"],
    }


def get_current_weather(city: str) -> dict[str, Any]:
    """Get current weather for a city from the free Open-Meteo API."""
    location = find_location(city)
    if location is None:
        return {"error": f"No location found for '{city}'."}

    weather_response = requests.get(
        FORECAST_API_URL,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "precipitation,weather_code,wind_speed_10m"
            ),
            "timezone": "auto",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    weather_response.raise_for_status()

    weather_data = weather_response.json()
    current = weather_data["current"]
    current_units = weather_data["current_units"]
    weather_code = current["weather_code"]

    return {
        "location": format_location(location),
        "observed_at": current["time"],
        "timezone": weather_data["timezone"],
        "conditions": WEATHER_DESCRIPTIONS.get(
            weather_code, f"unknown weather code {weather_code}"
        ),
        "temperature": {
            "value": current["temperature_2m"],
            "unit": current_units["temperature_2m"],
        },
        "apparent_temperature": {
            "value": current["apparent_temperature"],
            "unit": current_units["apparent_temperature"],
        },
        "relative_humidity": {
            "value": current["relative_humidity_2m"],
            "unit": current_units["relative_humidity_2m"],
        },
        "precipitation": {
            "value": current["precipitation"],
            "unit": current_units["precipitation"],
        },
        "wind_speed": {
            "value": current["wind_speed_10m"],
            "unit": current_units["wind_speed_10m"],
        },
        "source": "Open-Meteo",
    }


def get_aqi_category(aqi: float) -> str:
    if aqi <= 20:
        return "good"
    if aqi <= 40:
        return "fair"
    if aqi <= 60:
        return "moderate"
    if aqi <= 80:
        return "poor"
    if aqi <= 100:
        return "very poor"
    return "extremely poor"


def get_current_air_quality(city: str) -> dict[str, Any]:
    """Get current air quality for a city from the free Open-Meteo API."""
    location = find_location(city)
    if location is None:
        return {"error": f"No location found for '{city}'."}

    air_quality_response = requests.get(
        AIR_QUALITY_API_URL,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "european_aqi,pm10,pm2_5,nitrogen_dioxide,ozone",
            "timezone": "auto",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    air_quality_response.raise_for_status()

    air_quality_data = air_quality_response.json()
    current = air_quality_data["current"]
    current_units = air_quality_data["current_units"]
    european_aqi = current["european_aqi"]

    return {
        "location": format_location(location),
        "observed_at": current["time"],
        "timezone": air_quality_data["timezone"],
        "european_aqi": {
            "value": european_aqi,
            "unit": current_units["european_aqi"],
            "category": get_aqi_category(european_aqi),
        },
        "pm10": {"value": current["pm10"], "unit": current_units["pm10"]},
        "pm2_5": {"value": current["pm2_5"], "unit": current_units["pm2_5"]},
        "nitrogen_dioxide": {
            "value": current["nitrogen_dioxide"],
            "unit": current_units["nitrogen_dioxide"],
        },
        "ozone": {"value": current["ozone"], "unit": current_units["ozone"]},
        "source": "Open-Meteo and CAMS ENSEMBLE",
    }


WEATHER_TOOL_DECLARATION = {
    "name": "get_current_weather",
    "description": (
        "Gets real-time weather for a requested city from Open-Meteo. "
        "Use it whenever the user asks about current weather or whether current "
        "conditions are suitable for an outdoor activity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    "City name, optionally including a country, "
                    "e.g. Prague, Czechia."
                ),
            }
        },
        "required": ["city"],
    },
}

AIR_QUALITY_TOOL_DECLARATION = {
    "name": "get_current_air_quality",
    "description": (
        "Gets the current European Air Quality Index and pollutant levels for a "
        "requested city from Open-Meteo. Use it for questions about air pollution, "
        "health considerations, or suitability for outdoor activities."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    "City name, optionally including a country, "
                    "e.g. Prague, Czechia."
                ),
            }
        },
        "required": ["city"],
    },
}

AVAILABLE_FUNCTIONS = {
    "get_current_weather": get_current_weather,
    "get_current_air_quality": get_current_air_quality,
}


def get_content_label(content: types.Content) -> str:
    parts = content.parts or []
    if any(part.function_call for part in parts):
        return "model tool request"
    if any(part.function_response for part in parts):
        return "tool result"
    return f"{content.role} text"


def print_request_summary(
    contents: list[types.Content], call_number: int, allow_tool_calls: bool
) -> None:
    tool_mode = "AUTO (model decides)" if allow_tool_calls else "NONE (final answer)"
    history = " -> ".join(get_content_label(content) for content in contents)

    print(f"\n=== Model call {call_number}/{MAX_MODEL_CALLS} ===")
    print(f"Model: {MODEL}")
    print(f"Tool calling mode: {tool_mode}")
    print(f"Conversation items sent: {len(contents)}")
    print(f"Conversation history: {history}")
    print("Sending request to Gemini...")


def print_response_summary(response: types.GenerateContentResponse) -> None:
    candidate = response.candidates[0]
    tool_calls = response.function_calls or []
    response_type = (
        f"{len(tool_calls)} tool call(s)" if tool_calls else "final text"
    )
    finish_reason = (
        candidate.finish_reason.value if candidate.finish_reason else "UNKNOWN"
    )

    print("\n--- Model response ---")
    print(f"Response type: {response_type}")
    print(f"Finish reason: {finish_reason}")

    usage = response.usage_metadata
    if usage:
        token_parts = [
            f"input={usage.prompt_token_count or 0}",
            f"output={usage.candidates_token_count or 0}",
        ]
        if usage.thoughts_token_count:
            token_parts.append(f"thinking={usage.thoughts_token_count}")
        if usage.tool_use_prompt_token_count:
            token_parts.append(f"tool-context={usage.tool_use_prompt_token_count}")
        token_parts.append(f"total={usage.total_token_count or 0}")
        print(f"Tokens: {', '.join(token_parts)}")

    if tool_calls:
        print("Requested tools:")
        for index, tool_call in enumerate(tool_calls, start=1):
            print(f"  {index}. {tool_call.name}")
            print(f"     call id: {tool_call.id or 'not provided'}")
            print(
                "     arguments: "
                f"{json.dumps(dict(tool_call.args), ensure_ascii=False)}"
            )
    else:
        text_length = len(response.text or "")
        print(f"Text length: {text_length} characters")


def create_config(
    gemini_tools: types.Tool, allow_tool_calls: bool
) -> types.GenerateContentConfig:
    mode = "AUTO" if allow_tool_calls else "NONE"
    return types.GenerateContentConfig(
        system_instruction=(
            "You are a walking conditions assistant. Your job is to help the user "
            "decide whether current conditions are suitable for going for a walk. "
            "Answer in the same language as the user. If the user asks about current "
            "conditions and provides a location, use the weather and air-quality "
            "tools as needed. If a location is required but missing, ask for it "
            "instead of guessing. Evaluate precipitation, temperature, apparent "
            "temperature, wind, weather hazards, air quality, and local time when "
            "those data are available. Start with a clear recommendation: suitable, "
            "suitable with precautions, or not suitable. Then briefly explain the "
            "main reasons and suggest practical precautions such as clothing, "
            "visibility, or limiting exposure. Clearly distinguish measured data "
            "from your recommendation and never invent conditions not returned by "
            "a tool. You may answer directly when no tool is needed. Keep unrelated "
            "answers brief and remind the user that your specialty is assessing "
            "conditions for a walk."
        ),
        tools=[gemini_tools],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=mode)
        ),
    )


def execute_tool(tool_call: types.FunctionCall) -> dict[str, Any]:
    function_to_call = AVAILABLE_FUNCTIONS.get(tool_call.name)
    if function_to_call is None:
        return {"error": f"Unknown tool: {tool_call.name}"}

    try:
        return function_to_call(**tool_call.args)
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        return {"error": f"Tool execution failed: {error}"}


def ask_walking_assistant(question: str, client: genai.Client) -> str:
    gemini_tools = types.Tool(
        function_declarations=[
            WEATHER_TOOL_DECLARATION,
            AIR_QUALITY_TOOL_DECLARATION,
        ]
    )
    contents = [
        types.Content(role="user", parts=[types.Part(text=question)]),
    ]

    print("\n=== Agent run ===")
    print(f"Available tools: {', '.join(AVAILABLE_FUNCTIONS)}")
    print(f"Maximum model calls: {MAX_MODEL_CALLS}")

    for call_number in range(1, MAX_MODEL_CALLS + 1):
        # Reserve the final model call for a user-facing answer without more tools.
        allow_tool_calls = call_number < MAX_MODEL_CALLS
        print_request_summary(contents, call_number, allow_tool_calls)

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=create_config(gemini_tools, allow_tool_calls),
        )
        print_response_summary(response)
        contents.append(response.candidates[0].content)

        tool_calls = response.function_calls or []
        if not tool_calls:
            print("Agent decision: return the model's text to the user.")
            return response.text or "The model returned an empty response."

        if not allow_tool_calls:
            print("Agent stopped: the maximum number of model calls was reached.")
            return response.text or "The conversation reached its maximum length."

        function_responses = []
        for index, tool_call in enumerate(tool_calls, start=1):
            print(f"\n--- Tool execution {index}/{len(tool_calls)} ---")
            print(f"Executing: {tool_call.name}")
            print("The Python application, not Gemini, now runs the function.")
            tool_result = execute_tool(tool_call)
            print("Result returned by the tool:")
            print(json.dumps(tool_result, ensure_ascii=False, indent=2))
            function_responses.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=tool_call.id,
                        name=tool_call.name,
                        response={"result": tool_result},
                    )
                )
            )

        contents.append(types.Content(role="user", parts=function_responses))
        print(
            f"Added {len(function_responses)} tool result(s) to the conversation "
            "and continuing the loop."
        )

    return "The conversation reached its maximum length."


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    print("Hi! I help assess whether conditions are suitable for a walk.")
    question = input("Your question: ").strip()
    if not question:
        raise SystemExit("Question cannot be empty.")

    client = genai.Client(api_key=api_key)
    answer = ask_walking_assistant(question, client)
    print(f"\n=== Walking assistant answer ===\n{answer}")


if __name__ == "__main__":
    main()

import logging

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WeatherInput(BaseModel):
    location: str = Field(
        description="City name or location, e.g. 'London', 'New York'"
    )


def _get_weather_impl(location: str) -> dict:
    """Fetches current weather for a location."""
    logger.info(f"Fetching weather for: {location}")
    try:
        resp = requests.get(
            f"https://wttr.in/{requests.utils.quote(location)}",
            params={"format": "j1"},
            timeout=(5, 20),
        )
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        return {
            "city": city,
            "country": country,
            "condition": current["weatherDesc"][0]["value"],
            "temp_c": current["temp_C"],
            "temp_f": current["temp_F"],
            "feels_like_c": current["FeelsLikeC"],
            "feels_like_f": current["FeelsLikeF"],
            "humidity": current["humidity"],
            "wind_speed": current["windspeedKmph"],
        }
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return {"error": f"Could not fetch weather for '{location}': {e}"}


class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather conditions for a city or location"
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        result = _get_weather_impl(location)
        if "error" in result:
            return result["error"]

        return (
            f"Weather in {result['city']}, {result['country']}:\n"
            f"Condition: {result['condition']}\n"
            f"Temp: {result['temp_c']}°C / {result['temp_f']}°F\n"
            f"Feels like: {result['feels_like_c']}°C / {result['feels_like_f']}°F\n"
            f"Humidity: {result['humidity']}% | Wind: {result['wind_speed']} km/h"
        )

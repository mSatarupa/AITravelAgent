import os
import json
import requests
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# LangChain imports for 1.2.7 (new architecture)
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# =============================================================================
# TOOLS
# =============================================================================

@tool
def get_place_details(place_name: str, city: str) -> str:
    """
    Search for a tourist attraction by name and city, return full address and coordinates.
    
    Args:
        place_name: Name of attraction (e.g., "Eifel Tower")
        city: City name (e.g., "Paris")
    
    Returns:
        JSON with name, address, latitude, longitude
    """
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    query = f"{place_name}, {city}"
    
    params = {"query": query, "key": GOOGLE_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    
    if not data.get("results"):
        return json.dumps({"error": f"No results found for '{query}'"})
    
    place = data["results"][0]
    result = {
        "name": place["name"],
        "address": place["formatted_address"],
        "latitude": place["geometry"]["location"]["lat"],
        "longitude": place["geometry"]["location"]["lng"]
    }
    
    return json.dumps(result, indent=2)
@tool
def get_weather_forecast(latitude: float, longitude: float, date: str) -> str:
    """
    Get weather forecast for a specific location and date.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        date: Date in YYYY-MM-DD format
    
    Returns:
        JSON with weather, temperature, clothing recommendations
    """
    url = "https://weather.googleapis.com/v1/forecast/days:lookup"
    params = {
        "location.latitude": latitude,
        "location.longitude": longitude,
        "days": 4,
        "key": GOOGLE_API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code != 200:
        return json.dumps({"error": "Failed to fetch weather data"})
    
    for day in data.get("forecastDays", []):
        d = day["displayDate"]
        forecast_date = f"{d['year']}-{d['month']:02d}-{d['day']:02d}"
        
        if forecast_date == date:
            daytime = day["daytimeForecast"]
            max_temp = day["maxTemperature"]["degrees"]
            min_temp = day["minTemperature"]["degrees"]
            precip_prob = daytime["precipitation"]["probability"]["percent"]
            
            # Generate clothing recommendation
            avg_temp = (max_temp + min_temp) / 2
            if avg_temp < 0:
                clothing = "heavy winter coat, thermal layers, gloves, hat"
            elif avg_temp < 10:
                clothing = "warm jacket and sweater"
            elif avg_temp < 20:
                clothing = "light jacket or cardigan"
            else:
                clothing = "t-shirt and light clothing"
            
            if precip_prob > 50:
                clothing += ", waterproof jacket"
            elif precip_prob > 30:
                clothing += ", carry an umbrella"
            
            result = {
                "date": forecast_date,
                "condition": daytime["weatherCondition"]["description"]["text"],
                "max_temp_c": max_temp,
                "min_temp_c": min_temp,
                "feels_like_max_c": day["feelsLikeMaxTemperature"]["degrees"],
                "feels_like_min_c": day["feelsLikeMinTemperature"]["degrees"],
                "precipitation_probability": precip_prob,
                "wind_kph": daytime["wind"]["speed"]["value"],
                "umbrella_needed": "Yes" if precip_prob > 30 else "No",
                "clothing_recommendation": clothing
            }
            
            return json.dumps(result, indent=2)
    
    return json.dumps({"error": f"No forecast for {date}. Only next 4 days available."})


@tool
def get_air_quality(latitude: float, longitude: float) -> str:
    """
    Get air quality index (AQI) and mask recommendation.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    
    Returns:
        JSON with AQI, category, mask_needed
    """
    url = "https://airquality.googleapis.com/v1/currentConditions:lookup"
    payload = {"location": {"latitude": latitude, "longitude": longitude}}
    
    response = requests.post(
        f"{url}?key={GOOGLE_API_KEY}",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    data = response.json()
    
    if response.status_code != 200:
        return json.dumps({"error": "Failed to fetch air quality"})
    
    indexes = data.get("indexes", [])
    if not indexes:
        return json.dumps({"error": "No air quality data available"})
    
    aqi_value = indexes[0].get("aqi", 0)
    aqi_category = indexes[0].get("category", "Unknown")
    mask_needed = "Yes" if aqi_value > 100 else "No"
    
    result = {
        "aqi": aqi_value,
        "category": aqi_category,
        "mask_needed": mask_needed
    }
    
    return json.dumps(result, indent=2)


# =============================================================================
# AGENT CREATION (Using new LangChain 1.2.7 architecture)
# =============================================================================

def create_travel_agent():
    """
    Create agent using LangChain 1.2.7 - manual tool calling loop.
    
    Since LangChain 1.2.7's create_agent doesn't automatically execute tools,
    we'll use the LLM with bound tools and manually execute the tool loop.
    This is still using LangChain's built-in functions (bind_tools).
    """
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=GROQ_API_KEY
    )
    
    tools = [get_place_details, get_weather_forecast, get_air_quality]
    
    # Bind tools to LLM (LangChain's built-in function)
    llm_with_tools = llm.bind_tools(tools)
    
    # Create tool map for execution
    tool_map = {tool.name: tool for tool in tools}
    
    return llm_with_tools, tool_map


def run_agent_with_tools(llm_with_tools, tool_map, messages, max_iterations=15):
    """
    Execute the agent with tool calling loop.
    
    This manually implements the tool-calling loop that was automatic in older
    LangChain versions. It's using LangChain's built-in tool binding and execution.
    """
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Get LLM response
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if there are tool calls
        if not response.tool_calls:
            # No more tool calls - agent is done
            break
        
        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"Calling tool: {tool_name} with {tool_args}")
            
            # Execute the tool
            if tool_name in tool_map:
                tool_result = tool_map[tool_name].invoke(tool_args)
                
                # Add tool result to messages
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                ))
            else:
                print(f"Tool {tool_name} not found!")
    
    return messages


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the travel planning agent."""
    
    print("=" * 80)
    print("AI Travel Planning Agent (LangChain 1.2.7)")
    print("=" * 80)
    print()
    
    llm_with_tools, tool_map = create_travel_agent()
    
    # System prompt
    system_message = """You are an AI travel planning assistant.

WORKFLOW:
1. For EACH attraction, use get_place_details(place_name, city) to find address and coordinates
2. For EACH city, use get_weather_forecast(latitude, longitude, date) for weather
3. For EACH city, use get_air_quality(latitude, longitude) for air quality
4. Create comprehensive travel report

RULES:
- ALWAYS call get_place_details for EVERY attraction
- Weather only works for next 4 days
- Umbrella needed if precipitation > 30%
- Mask needed if AQI > 100
- Calculate TOTAL masks needed
- ONE clear summary (no repetition)

OUTPUT FORMAT:
=== Here Is the AI assisted TRAVEL PLAN for you ===

CITY 1: [City] - [Date]
Attractions:
1. [Name] - [Address] ([Time])

Weather: [Condition], [Temp]°C
Clothing: [Recommendations]
Umbrella: [Yes/No]
Air Quality: AQI [Number] - [Category]
Mask Needed: [Yes/No]

SUMMARY:
Total Masks Needed: [Number]"""
    
    # UPDATE THESE DATES TO BE WITHIN NEXT 4 DAYS!
    travel_input = """
Please create a detailed travel plan for:

City1: Paris 2026-02-16
Eiffel Tower;8am-9am
Louvre Museum;10am-11am
Arc de Triomphe;12pm-1pm

City2: Brussels 2026-02-18
Atonium;9am-11am
Grand Place;12pm-1pm
"""
    
    print("TRAVEL REQUEST:")
    print(travel_input)
    print("\n" + "=" * 80 + "\n")
    
    # Create initial messages
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=travel_input)
    ]
    
    # Run agent with tool calling loop
    messages = run_agent_with_tools(llm_with_tools, tool_map, messages)
    
    # Extract final answer
    final_message = messages[-1].content
    
    print("\nTRAVEL PLAN:")
    print(final_message)
    print("\n" + "=" * 80 + "\n")
    
    # Interactive mode with memory
    print("Ask follow-up questions (type 'quit' to exit):")
    print("-" * 80)
    
    while True:
        user_input = input("\nSataM: ").strip()
        
        if user_input.lower() in {"q", "quit", "exit"}:
            print("\nThank you! Have a wonderful trip!")
            break
        
        if not user_input:
            continue
        
        # Add user message
        messages.append(HumanMessage(content=user_input))
        
        # Run agent with tools
        messages = run_agent_with_tools(llm_with_tools, tool_map, messages)
        
        # Get answer
        answer = messages[-1].content
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()



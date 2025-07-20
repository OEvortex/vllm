import json
import requests
from typing import Dict, List, Any, Optional

class OpenAIToolClient:
    def __init__(self, base_url: str = "https://8082-01k0k9r9rbaz3d0p3vcrdykgvg.cloudspaces.litng.ai/v1", api_key: str = "dummy"):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def get_weather_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }

    def get_calculator_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate, e.g. '2 + 3 * 4'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "get_weather":
            location = arguments.get("location", "Unknown")
            unit = arguments.get("unit", "celsius")
            return f"The weather in {location} is 22°{'C' if unit == 'celsius' else 'F'} and sunny."
        elif tool_name == "calculate":
            try:
                expression = arguments.get("expression", "0")
                result = eval(expression)
                return f"The result of {expression} is {result}"
            except Exception as e:
                return f"Error calculating {expression}: {str(e)}"
        return f"Unknown tool: {tool_name}"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: str = "HelpingAI/Dhanishtha-2.0-preview",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        print("📝 Making non-streaming request...")
        print(f"📝 Payload: {json.dumps(payload, indent=2)}")
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error in non-streaming request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"📄 Response content: {e.response.text}")
            raise

    def handle_tool_calls(self, message: Dict[str, Any]) -> List[Dict[str, str]]:
        tool_messages = []
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_id = tool_call["id"]
                print(f"🔧 Executing tool: {tool_name} with args: {tool_args}")
                result = self.execute_tool(tool_name, tool_args)
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result
                })
                print(f"✅ Tool result: {result}")
        return tool_messages

def demo_non_streaming():
    print("=" * 60)
    print("💧 NON-STREAMING TOOL CALLING DEMO")
    print("=" * 60)
    client = OpenAIToolClient()
    tools = [
        client.get_weather_tool(),
        client.get_calculator_tool()
    ]
    messages = [
        {"role": "user", "content": "Can you tell me the weather in London and also calculate the square root of 144?"}
    ]
    try:
        response = client.chat_completion(messages, tools)
        print("\n🤖 Assistant response:")
        assistant_message = response["choices"][0]["message"]
        print(assistant_message.get("content", ""))
        if "tool_calls" in assistant_message:
            tool_messages = client.handle_tool_calls(assistant_message)
            if tool_messages:
                messages.append(assistant_message)
                messages.extend(tool_messages)
                print("\n🔄 Getting follow-up response with tool results:")
                followup = client.chat_completion(messages, tools)
                followup_message = followup["choices"][0]["message"]
                print(followup_message.get("content", ""))
        print("\n" + "=" * 60)
        print("✨ Non-streaming demo completed! Check the outputs above.")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Error in non-streaming demo: {e}")

if __name__ == "__main__":
    print("🤖 OpenAI-Compatible Non-Streaming Tool Calling Demo")
    print("Server URL: https://8082-01k0k9r9rbaz3d0p3vcrdykgvg.cloudspaces.litng.ai/v1")
    demo_non_streaming()

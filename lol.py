import json
import requests
import time
from typing import Dict, List, Any, Optional

class OpenAIToolClient:
    def __init__(self, base_url: str = "https://8082-01k0k9r9rbaz3d0p3vcrdykgvg.cloudspaces.litng.ai/v1", api_key: str = "dummy"):
        """
        Initialize the OpenAI-compatible client for your vLLM server
        
        Args:
            base_url: Base URL of your vLLM server (default: your cloudspace URL)
            api_key: API key (can be dummy for local servers)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def get_weather_tool(self) -> Dict[str, Any]:
        """Example weather tool definition"""
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
        """Example calculator tool definition"""
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
        """
        Mock tool execution - replace with your actual tool implementations
        """
        if tool_name == "get_weather":
            location = arguments.get("location", "Unknown")
            unit = arguments.get("unit", "celsius")
            return f"The weather in {location} is 22°{'C' if unit == 'celsius' else 'F'} and sunny."
        
        elif tool_name == "calculate":
            try:
                expression = arguments.get("expression", "0")
                # Simple eval - in production, use a safer math parser
                result = eval(expression)
                return f"The result of {expression} is {result}"
            except Exception as e:
                return f"Error calculating {expression}: {str(e)}"
        
        return f"Unknown tool: {tool_name}"
    
    def chat_completion_streaming(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        model: str = "HelpingAI/Dhanishtha-2.0-preview",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Streaming chat completion with tool calling
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        print("🌊 Making streaming request...")
        print(f"📝 Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            continue
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Error in streaming request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"📄 Response content: {e.response.text}")
            raise
    
    def handle_tool_calls(self, message: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Handle tool calls from the assistant's response
        """
        tool_messages = []
        
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_id = tool_call["id"]
                
                print(f"🔧 Executing tool: {tool_name} with args: {tool_args}")
                
                # Execute the tool
                result = self.execute_tool(tool_name, tool_args)
                
                # Add tool result message
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result
                })
                
                print(f"✅ Tool result: {result}")
        
        return tool_messages

def demo_streaming():
    """Demonstrate streaming tool calling"""
    print("=" * 60)
    print("🌊 STREAMING TOOL CALLING DEMO")
    print("=" * 60)
    
    client = OpenAIToolClient()
    
    # Define available tools
    tools = [
        client.get_weather_tool(),
        client.get_calculator_tool()
    ]
    
    # Initial conversation
    messages = [
        {"role": "user", "content": "Can you tell me the weather in London and also calculate the square root of 144?"}
    ]
    
    try:
        print("🌊 Streaming response:")
        
        # Collect streaming response
        assistant_message = {"role": "assistant", "content": "", "tool_calls": []}
        current_tool_call = None
        
        for chunk in client.chat_completion_streaming(messages, tools):
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                
                # Handle content
                if "content" in delta and delta["content"]:
                    print(delta["content"], end="", flush=True)
                    assistant_message["content"] += delta["content"]
                
                # Handle tool calls
                if "tool_calls" in delta:
                    for tool_call_delta in delta["tool_calls"]:
                        index = tool_call_delta.get("index", 0)
                        
                        # Ensure we have enough tool calls in our list
                        while len(assistant_message["tool_calls"]) <= index:
                            assistant_message["tool_calls"].append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        if "id" in tool_call_delta:
                            assistant_message["tool_calls"][index]["id"] = tool_call_delta["id"]
                        
                        if "function" in tool_call_delta:
                            func_delta = tool_call_delta["function"]
                            if "name" in func_delta:
                                assistant_message["tool_calls"][index]["function"]["name"] += func_delta["name"]
                            if "arguments" in func_delta:
                                assistant_message["tool_calls"][index]["function"]["arguments"] += func_delta["arguments"]
        
        print("\n")  # New line after streaming content
        
        # Clean up empty tool calls
        assistant_message["tool_calls"] = [tc for tc in assistant_message["tool_calls"] if tc["id"]]
        if not assistant_message["tool_calls"]:
            del assistant_message["tool_calls"]
        
        messages.append(assistant_message)
        
        # Handle tool calls if any
        if "tool_calls" in assistant_message:
            tool_messages = client.handle_tool_calls(assistant_message)
            if tool_messages:
                messages.extend(tool_messages)
                
                # Stream the final response
                print("\n🔄 Streaming follow-up response with tool results:")
                for chunk in client.chat_completion_streaming(messages, tools):
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            print(delta["content"], end="", flush=True)
                
                print("\n")  # New line after final streaming
    
    except Exception as e:
        print(f"❌ Error in streaming demo: {e}")

def test_server_connection():
    """Test if the server is running"""
    print("🔍 Testing server connection...")
    
    try:
        client = OpenAIToolClient()
        response = requests.get(f"{client.base_url.replace('/v1', '')}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Server is running!")
            return True
        else:
            print(f"⚠️ Server responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure your vLLM server is running on your cloudspace")
        return False

if __name__ == "__main__":
    print("🤖 OpenAI-Compatible Streaming Tool Calling Demo")
    print("Server URL: https://8082-01k0k9r9rbaz3d0p3vcrdykgvg.cloudspaces.litng.ai/v1")
    
    if not test_server_connection():
        print("\n💡 Make sure your vLLM server is running with tool calling enabled")
        exit(1)
    
    # Run streaming demo
    demo_streaming()
    
    print("\n" + "=" * 60)
    print("✨ Streaming demo completed! Check the outputs above.")
    print("=" * 60)

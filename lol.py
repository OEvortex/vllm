
# Simple streaming tool calling example
import json
import requests

def get_calculator_tool():
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

def stream_tool_call():
    url = "https://8082-01k0k9r9rbaz3d0p3vcrdykgvg.cloudspaces.litng.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy"
    }
    tools = [get_calculator_tool()]
    messages = [
        {"role": "user", "content": "What is 12 * 12 + 7?"}
    ]
    payload = {
        "model": "HelpingAI/Dhanishtha-2.0-preview",
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": True,
        "tools": tools,
        "tool_choice": "auto"
    }
    print("🌊 Streaming response:")
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    response.raise_for_status()
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data.strip() == '[DONE]':
                    break
                chunk = json.loads(data)
                print(chunk)
    print("\n")

if __name__ == "__main__":
    stream_tool_call()

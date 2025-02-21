from flask import Flask, request, jsonify, json
import logging
import requests
import os
from database import APIKeyManager
from datetime import datetime, timedelta
import uuid
import base64
import litellm
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize configs
LARK_APP_ID = os.getenv("LARK_APP_ID")
LARK_APP_SECRET = os.getenv("LARK_APP_SECRET")
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# Initialize database
db = APIKeyManager()

def get_tenant_token():
    """Get Lark tenant access token"""
    url = "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal/"
    post_data = {
        "app_id": LARK_APP_ID,
        "app_secret": LARK_APP_SECRET
    }
    r = requests.post(url, json=post_data)
    return r.json()["tenant_access_token"]

def send_message(user_id: str, content: str, msg_type: str = "text"):
    """Send message to user"""
    try:
        token = get_tenant_token()
        url = "https://open.larksuite.com/open-apis/im/v1/messages"
        params = {"receive_id_type": "open_id"}
        
        if msg_type == "text":
            req = {
                "receive_id": user_id,
                "msg_type": "text",
                "content": json.dumps({"text": content})
            }
        else:  # post type for rich text
            msg_content = {
                "zh_cn": {
                    "title": "Message",
                    "content": [[{
                        "tag": "text",
                        "text": content
                    }]]
                }
            }
            req = {
                "receive_id": user_id,
                "msg_type": "post",
                "content": json.dumps(msg_content)
            }

        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, params=params, headers=headers, json=req)
        logger.debug(f"Send message response: {response.json()}")
        
    except Exception as e:
        logger.error(f"Error sending message: {e}")

def generate_litellm_key(user_id: str) -> str:
    """Generate a unique LiteLLM proxy key using proxy server API"""
    try:
        master_key = os.getenv("LITELLM_MASTER_KEY")
        proxy_url = os.getenv("LITELLM_PROXY_URL")
        
        # 生成唯一的密钥别名
        key_alias = f"user-{user_id}-{int(datetime.now().timestamp())}"  # 添加时间戳确保唯一性
        
        # 生成密钥请求
        response = requests.post(
            f"{proxy_url}/key/generate",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json={
                "key_alias": key_alias,
                "team_id": "larkbot",
                "duration": None,
                "models": ["*"],  # 允许访问所有模型
                "spend": 0,
                "max_parallel_requests": 5,
                "metadata": {
                    "user_id": user_id,
                    "created_by": "larkbot"
                }
            }
        )
        
        if response.status_code == 200:
            key_data = response.json()
            logger.info(f"Generated key response: {response.text}")
            return key_data["key"]
        else:
            logger.error(f"Failed to generate key: {response.text}")
            raise Exception("Failed to generate key")
            
    except Exception as e:
        logger.error(f"Error generating key: {e}")
        raise

def revoke_litellm_key(key: str) -> bool:
    """Revoke a LiteLLM proxy key using proxy server API"""
    try:
        master_key = os.getenv("LITELLM_MASTER_KEY")
        proxy_url = os.getenv("LITELLM_PROXY_URL")
        
        # 调用代理服务器的 key 删除 API
        response = requests.post(
            f"{proxy_url}/key/delete",  # 改为 /key/delete
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json={
                "keys": [key]  # 改为列表格式
            }
        )
        
        logger.info(f"Revoke key response: {response.status_code} - {response.text}")
        return response.status_code == 200
            
    except Exception as e:
        logger.error(f"Error revoking key: {e}")
        return False

def get_available_models():
    """Get list of available models from LiteLLM proxy"""
    try:
        master_key = os.getenv("LITELLM_MASTER_KEY")
        proxy_url = os.getenv("LITELLM_PROXY_URL")
        
        response = requests.get(
            f"{proxy_url}/models",
            headers={"Authorization": f"Bearer {master_key}"}
        )
        if response.status_code == 200:
            # 只返回配置的模型名称（id），而不是原始模型名称
            return [model["id"] for model in response.json()["data"]]
        return []
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return []

def get_model_info(detailed=False):
    """Get information about available models from LiteLLM proxy"""
    try:
        master_key = os.getenv("LITELLM_MASTER_KEY")
        proxy_url = os.getenv("LITELLM_PROXY_URL")
        
        response = requests.get(
            f"{proxy_url}/models",
            headers={"Authorization": f"Bearer {master_key}"}
        )
        
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            if detailed:
                # 详细信息（用于 /help）
                models_info = []
                for model in models_data:
                    model_id = model.get("id", "")
                    model_name = model.get("model_name", "")
                    context_length = model.get("context_length", "Unknown")
                    pricing = model.get("pricing", {})
                    
                    info = f"• {model_id}"
                    if model_name and model_name != model_id:
                        info += f" (Base: {model_name})"
                    if context_length != "Unknown":
                        info += f" - {context_length} tokens"
                    if pricing:
                        input_price = pricing.get("input_price_per_token", 0)
                        output_price = pricing.get("output_price_per_token", 0)
                        if input_price or output_price:
                            info += f"\n  Pricing: ${input_price}/1K input, ${output_price}/1K output"
                    
                    models_info.append(info)
            else:
                # 简洁信息（用于 /list）
                models_info = []
                for model in models_data:
                    model_id = model.get("id", "")
                    model_name = model.get("model_name", "")
                    info = f"• {model_id}"
                    if model_name and model_name != model_id:
                        info += f"\n  Original: {model_name}"
                    models_info.append(info)
            
            return models_info
        return []
    except Exception as e:
        logger.error(f"Error getting models info: {e}")
        return []

def handle_command(user_id: str, command: str):
    """Handle bot commands"""
    parts = command.split()
    cmd = parts[0].lower()

    if cmd == "/help":
        message = (
            "**Available Commands:**\n"
            "• `/create` - Create a new LiteLLM proxy key\n"
            "• `/show` - Show your proxy key and available models\n"
            "• `/revoke` - Revoke your current proxy key\n"
            "• `/usage` - Show your API usage statistics\n"
            "• `/chat model=<model_name>` - Set your default chat model\n"
            "• `/help` - Show this help message\n\n"
            "Just type your message directly to chat with the AI!"
        )
        send_message(user_id, message, "post")

    elif cmd == "/chat":
        if len(parts) < 2 or not parts[1].startswith("model="):
            # 获取简洁的模型信息用于显示
            models_info = get_model_info(detailed=False)
            message = (
                "Usage: /chat model=<model_name>\n\n"
                "Available models:\n"
                f"{chr(10).join(models_info)}\n\n"
                "Note: Use the model name (not the Original name) shown above."
            )
            send_message(user_id, message, "post")
            return
            
        model = parts[1].split("=")[1]
        available_models = get_available_models()
        if model not in available_models:
            models_info = get_model_info(detailed=False)
            message = (
                "Invalid model name. Please choose from:\n"
                f"{chr(10).join(models_info)}\n\n"
                "Note: Use the model name (not the Original name) shown above."
            )
            send_message(user_id, message, "post")
            return
            
        db.set_chat_model(user_id, model)
        send_message(user_id, f"Default chat model set to: {model}", "post")

    elif cmd == "/create":
        # Check if user already has a key
        if db.has_active_key(user_id, "litellm"):
            send_message(user_id, "You already have an active key. Use /revoke to revoke it first.", "post")
            return
            
        new_key = generate_litellm_key(user_id)
        db.add_api_key(user_id, "litellm", new_key)
        key_text = f"""
**Your New LiteLLM Proxy Key:**
```
{new_key}
```

**How to use this key:**
1. Use this key as your API key when making requests
2. The proxy supports OpenAI-compatible API format
3. Available models are configured by your administrator

Note: You can only have one active key at a time.
        """
        send_message(user_id, key_text, "post")

    elif cmd == "/revoke":
        if not db.has_active_key(user_id, "litellm"):
            send_message(user_id, "You don't have any active key to revoke.", "post")
            return
            
        # 获取当前密钥
        current_key = db.get_api_key(user_id, "litellm")
        
        # 通过 API 撤销密钥
        if revoke_litellm_key(current_key):
            # 从数据库中删除密钥
            db.revoke_key(user_id, "litellm")
            send_message(user_id, "Your key has been revoked successfully.", "post")
        else:
            send_message(user_id, "Failed to revoke key. Please try again or contact administrator.", "post")

    elif cmd == "/show":
        try:
            # 获取用户的 API 密钥
            key = db.get_api_key(user_id, "litellm")
            key_text = ""
            if key:
                key_text = (
                    "**Your Current LiteLLM Proxy Key:**\n"
                    "```\n"
                    f"{key}\n"
                    "```\n"
                )
            else:
                key_text = "You don't have a proxy key yet. Use /create to generate one.\n"

            # 获取模型信息
            models_response = requests.get(
                f"{LITELLM_PROXY_URL}/models",
                headers={"Authorization": f"Bearer {os.getenv('LITELLM_MASTER_KEY')}"}
            )
            
            if models_response.status_code != 200:
                raise Exception(f"Failed to get models information: {models_response.text}")
            
            # 获取当前用户的聊天模型
            current_model = db.get_chat_model(user_id)
            
            # 构建模型信息列表
            models_info = []
            for model in models_response.json().get("data", []):
                model_id = model.get("id", "")
                model_name = model.get("model_name", "")
                
                # 标记当前使用的模型
                current_marker = "✓ " if model_id == current_model else "  "
                
                # 构建模型信息字符串
                info = f"{current_marker}• {model_id}"
                if model_name:
                    info += f"\n    Original: {model_name}"
                
                models_info.append(info)
            
            message = (
                f"{key_text}\n"
                "**Available Models:**\n"
                f"{chr(10).join(models_info)}\n\n"
                "Use `/chat model=<model_name>` to change your chat model.\n"
                "Note: Use the model name (not the Original name) when setting the chat model."
            )
            
            send_message(user_id, message, "post")
            
        except Exception as e:
            logger.error(f"Error getting information: {str(e)}")
            send_message(user_id, f"Error: Failed to get information. Please try again later.", "post")

    elif cmd == "/usage":
        key = db.get_api_key(user_id, "litellm")
        if not key:
            send_message(user_id, "You don't have a proxy key yet. Use /create to generate one.")
            return
            
        try:
            # Get usage for last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            litellm.api_key = key
            usage = litellm.get_usage_metrics(
                start_date=start_date,
                end_date=end_date
            )
            
            # Build model stats string
            model_stats = []
            for model, stats in usage.model_stats.items():
                model_stats.append(f"• {model}:")
                model_stats.append(f"  - Requests: {stats['requests']:,}")
                model_stats.append(f"  - Tokens: {stats['total_tokens']:,}")
                model_stats.append(f"  - Cost: ${stats['total_cost']:.2f}")
            
            model_stats_str = "\n".join(model_stats)
            
            usage_text = f"""
**Your API Usage (Last 30 Days)**

📊 **Overview:**
• Total Requests: {usage.total_requests:,}
• Total Tokens: {usage.total_tokens:,}
• Estimated Cost: ${usage.total_cost:.2f}

📈 **By Model:**
{model_stats_str}

_Note: Usage data is updated every few minutes._
            """
            send_message(user_id, usage_text, "post")
            
        except Exception as e:
            send_message(user_id, f"Error fetching usage statistics. Please try again later or contact administrator if the issue persists.", "post")

    elif cmd == "/list":
        try:
            # 获取模型信息
            models_response = requests.get(
                f"{LITELLM_PROXY_URL}/models",
                headers={"Authorization": f"Bearer {os.getenv('LITELLM_MASTER_KEY')}"}
            )
            
            # 添加调试日志
            logger.info(f"Models API response status: {models_response.status_code}")
            logger.info(f"Models API response: {models_response.text}")
            
            if models_response.status_code != 200:
                raise Exception(f"Failed to get models information: {models_response.text}")
            
            # 获取当前用户的聊天模型
            current_model = db.get_chat_model(user_id)
            logger.info(f"Current model for user {user_id}: {current_model}")
            
            # 构建模型信息列表
            models_info = []
            for model in models_response.json().get("data", []):
                model_id = model.get("id", "")
                model_name = model.get("model_name", "")
                logger.info(f"Processing model: id={model_id}, name={model_name}")
                
                # 标记当前使用的模型
                current_marker = "✓ " if model_id == current_model else "  "
                
                # 构建模型信息字符串
                info = f"{current_marker}• {model_id}"
                if model_name:
                    info += f"\n    Original: {model_name}"
                
                models_info.append(info)
                logger.info(f"Added model info: {info}")
            
            message = (
                "**Available Models:**\n"
                f"{chr(10).join(models_info)}\n\n"
                "Use `/chat model=<model_name>` to change your chat model.\n"
                "Note: Use the model name (not the Original name) when setting the chat model."
            )
            
            logger.info(f"Final message: {message}")
            send_message(user_id, message, "post")
            
        except Exception as e:
            logger.error(f"Error getting models info: {str(e)}")
            send_message(user_id, f"Error: Failed to get models information. Please try again later.", "post")

    else:
        send_message(user_id, "Unknown command. Type /help for available commands.", "post")

@app.route("/query/message", methods=["POST"])
def handle_event():
    # Get request data
    data = request.get_json()
    logger.debug(f"Received event: {data}")

    # Handle challenge verification
    if "challenge" in data:
        return jsonify({"challenge": data.get("challenge")})

    try:
        header = data.get("header", {})
        event_type = header.get("event_type")

        if event_type == "im.message.receive_v1":
            event = data.get("event", {})
            message = event.get("message", {})
            
            if message.get("message_type") != "text":
                return jsonify({"status": "ok"})
                
            content = json.loads(message.get("content", "{}"))
            text = content.get("text", "").strip()
            sender_id = event.get("sender", {}).get("sender_id", {}).get("open_id")

            if text.startswith("/"):
                handle_command(sender_id, text)
                return jsonify({"status": "ok"})

            # 处理普通聊天消息
            key = db.get_api_key(sender_id, "litellm")
            if not key:
                send_message(sender_id, "You don't have a proxy key yet. Use /create to generate one.", "post")
                return jsonify({"status": "ok"})
            
            try:
                model = db.get_chat_model(sender_id)
                client = openai.OpenAI(
                    api_key=key,
                    base_url=LITELLM_PROXY_URL
                )
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=MAX_TOKENS
                )
                
                answer = response.choices[0].message.content
                send_message(sender_id, answer, "post")
                return jsonify({"status": "ok"})
                
            except Exception as e:
                logger.error(f"Error in chat: {str(e)}")
                error_message = str(e)
                
                # 根据错误类型返回更详细的错误信息
                if "credit balance is too low" in error_message.lower():
                    error_text = "Error: Your credit balance is too low."
                elif "rate limit" in error_message.lower():
                    error_text = "Error: Rate limit exceeded. Please try again later."
                elif "context length" in error_message.lower():
                    error_text = "Error: Input text is too long for this model."
                else:
                    error_text = f"Error: {error_message}"  # 返回原始错误信息
                
                send_message(sender_id, error_text, "post")
                return jsonify({"status": "error", "message": error_message})

    except Exception as e:
        logger.error(f"Error handling event: {e}")
        return jsonify({"status": "error", "message": str(e)})

    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 
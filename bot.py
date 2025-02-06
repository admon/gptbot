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
        
        # 获取可用模型列表
        models_response = requests.get(
            f"{proxy_url}/models",
            headers={"Authorization": f"Bearer {master_key}"}
        )
        if models_response.status_code != 200:
            logger.error(f"Failed to get models: {models_response.text}")
            raise Exception("Failed to get available models")
            
        available_models = []
        for model in models_response.json().get("data", []):
            model_id = model.get("id")
            if model_id:
                available_models.append(model_id)
                # 如果模型有别名，也添加到允许列表中
                aliases = model.get("aliases", [])
                available_models.extend(aliases)
        
        logger.info(f"Available models with aliases: {available_models}")
        
        # 生成密钥请求
        response = requests.post(
            f"{proxy_url}/key/generate",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json"
            },
            json={
                "key_alias": f"user-{user_id}",
                "team_id": "larkbot",
                "duration": None,
                "models": available_models,  # 包含所有模型及其别名
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
            return [model["id"] for model in response.json()["data"]]
        return []
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return []

def handle_command(user_id: str, command: str):
    """Handle bot commands"""
    parts = command.split()
    cmd = parts[0].lower()

    if cmd == "/help":
        available_models = get_available_models()
        models_text = "\n".join([f"• {model}" for model in available_models])
        
        help_text = f"""
**Available Commands:**
• `/create` - Create a new LiteLLM proxy key
• `/show` - Show your current LiteLLM proxy key
• `/revoke` - Revoke your current proxy key
• `/usage` - Show your API usage statistics
• `/ask <question>` - Ask a question to the AI model
• `/help` - Show this help message

**Available Models:**
{models_text}

This bot helps you manage your LiteLLM proxy key and interact with AI models.
Use any available model with the /ask command by specifying the model name.
        """
        send_message(user_id, help_text, "post")

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
        key = db.get_api_key(user_id, "litellm")
        if key:
            key_text = f"""
**Your Current LiteLLM Proxy Key:**
```
{key}
```

Note: This is your only active key. Use /revoke to generate a new one if needed.
            """
            send_message(user_id, key_text, "post")
        else:
            send_message(user_id, "You don't have a proxy key yet. Use /create to generate one.", "post")

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

    elif cmd == "/ask":
        if len(parts) < 2:
            send_message(user_id, "Usage: /ask [model=<model_name>] <your question>", "post")
            return
            
        # Check if model is specified
        if parts[1].startswith("model="):
            model = parts[1].split("=")[1]
            question = " ".join(parts[2:])
        else:
            model = DEFAULT_MODEL
            question = " ".join(parts[1:])
            
        key = db.get_api_key(user_id, "litellm")
        if not key:
            send_message(user_id, "You don't have a proxy key yet. Use /create to generate one.", "post")
            return
            
        try:
            litellm.api_key = key
            
            # 根据不同模型调整请求格式
            messages = []
            if "deepseek" in model.lower():
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant that provides accurate and helpful responses."},
                    {"role": "user", "content": question}
                ]
                # 添加 Deepseek 特定参数
                model_params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "api_base": LITELLM_PROXY_URL,
                    "temperature": 0.7,
                    "response_format": {"type": "text"}  # 确保返回文本格式
                }
            else:
                # 默认消息格式（适用于 OpenAI 等模型）
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": question}
                ]
                model_params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "api_base": LITELLM_PROXY_URL
                }
            
            # 发送请求
            response = litellm.completion(**model_params)
            
            # 处理响应
            answer = response.choices[0].message.content
            message_text = f"""
**Model:** {model}
**Question:** {question}

**Answer:** {answer}
            """
            send_message(user_id, message_text, "post")
            
        except litellm.AuthenticationError as e:
            logger.error(f"Authentication error: {str(e)}")
            send_message(user_id, "Authentication failed. Please use /create to generate a new API key.", "post")
        except litellm.BadRequestError as e:
            logger.error(f"Bad request error: {str(e)}")
            send_message(user_id, f"Error: The model {model} returned an error. Try another model or rephrase your question.", "post")
        except Exception as e:
            logger.error(f"Error in /ask command: {str(e)}")
            send_message(user_id, f"Error: {str(e)}", "post")

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

    # Handle message events
    try:
        header = data.get("header", {})
        event_type = header.get("event_type")

        if event_type == "im.message.receive_v1":
            event = data.get("event", {})
            message = event.get("message", {})
            
            if message.get("message_type") == "text":
                content = json.loads(message.get("content", "{}"))
                text = content.get("text", "").strip()
                sender_id = event.get("sender", {}).get("sender_id", {}).get("open_id")

                if text.startswith("/"):
                    handle_command(sender_id, text)

    except Exception as e:
        logger.error(f"Error handling event: {e}")

    return "", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 
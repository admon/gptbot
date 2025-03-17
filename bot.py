from flask import Flask, request, jsonify, json, abort
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
from cachetools import TTLCache
import hashlib
import hmac
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also log Flask's werkzeug at DEBUG level
logging.getLogger('werkzeug').setLevel(logging.DEBUG)

# Initialize configs
LARK_APP_ID = os.getenv("LARK_APP_ID")
LARK_APP_SECRET = os.getenv("LARK_APP_SECRET")
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# Lark Event Subscription configs
ENCRYPT_KEY = os.getenv("ENCRYPT_KEY", "").strip()
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")

# Validate required configs
if not ENCRYPT_KEY:
    logger.error("ENCRYPT_KEY is not set in environment variables")
    raise ValueError("ENCRYPT_KEY is required")

# Initialize database and message cache
db = APIKeyManager()
# Cache to store processed message IDs (20 minute TTL)
message_cache = TTLCache(maxsize=1000, ttl=1200)

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

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt data using AES-CBC according to Lark's specification"""
    try:
        logger.debug(f"Starting decryption of data: {encrypted_data[:20]}...")
        logger.debug(f"Using ENCRYPT_KEY (first 4 chars): {ENCRYPT_KEY[:4]}...")
        
        # Base64 decode the encrypted data
        encrypted_bytes = base64.b64decode(encrypted_data)
        logger.debug(f"Base64 decoded length: {len(encrypted_bytes)}")
        
        # Generate AES key by taking SHA256 hash of ENCRYPT_KEY
        key = hashlib.sha256(ENCRYPT_KEY.encode('utf-8')).digest()
        logger.debug(f"SHA256 key (hex): {key.hex()[:16]}...")
        
        # Use first 16 bytes of encrypted data as IV
        iv = encrypted_bytes[:16]
        # Get the actual encrypted content
        content = encrypted_bytes[16:]
        
        logger.debug(f"Decryption key length: {len(key)}")
        logger.debug(f"IV length: {len(iv)}")
        logger.debug(f"Content length: {len(content)}")
        logger.debug(f"IV (hex): {iv.hex()[:16]}...")
        
        # Create AES cipher in CBC mode
        cipher = AES.new(key, AES.MODE_CBC, iv)
        # Decrypt and unpad using PKCS7
        decrypted = unpad(cipher.decrypt(content), AES.block_size)
        decrypted_text = decrypted.decode('utf-8')
        logger.debug(f"Successfully decrypted to: {decrypted_text[:100]}")
        return decrypted_text
    except Exception as e:
        logger.error(f"Error decrypting data: {str(e)}")
        raise

def verify_signature(timestamp: str, nonce: str, body: bytes, signature: str) -> bool:
    """Verify the signature of Lark event according to Lark documentation
    
    Args:
        timestamp: X-Lark-Request-Timestamp header
        nonce: X-Lark-Request-Nonce header
        body: Raw request body bytes
        signature: X-Lark-Signature header
    """
    try:
        if not all([timestamp, nonce, body, signature, ENCRYPT_KEY]):
            logger.error("Missing required parameters for signature verification")
            return False
            
        logger.debug("=== Signature Verification Debug ===")
        logger.debug(f"Timestamp: {timestamp}")
        logger.debug(f"Nonce: {nonce}")
        logger.debug(f"Body length: {len(body)}")
        logger.debug(f"Raw Signature: {signature}")
        logger.debug(f"Encrypt Key length: {len(ENCRYPT_KEY)}")
        
        # Remove 'sha256=' prefix if present
        if signature.startswith('sha256='):
            signature = signature[7:]
            logger.debug(f"Cleaned Signature: {signature}")
        
        # Construct the string to sign according to Lark docs
        # Format: timestamp + nonce + encrypt_key + body
        key_bytes = ENCRYPT_KEY.encode('utf-8')
        timestamp_bytes = timestamp.encode('utf-8')
        nonce_bytes = nonce.encode('utf-8')
        
        # Join all components in bytes
        string_to_sign = b''.join([timestamp_bytes, nonce_bytes, key_bytes, body])
        logger.debug(f"String to sign length: {len(string_to_sign)}")
        
        # Calculate signature using SHA256
        calculated_signature = hashlib.sha256(string_to_sign).hexdigest()
        
        logger.debug(f"Calculated signature: {calculated_signature}")
        logger.debug(f"Received signature: {signature}")
        logger.debug("=== End Debug ===")
        
        return hmac.compare_digest(calculated_signature, signature)
    except Exception as e:
        logger.error(f"Error in signature verification: {str(e)}")
        return False

@app.route("/query/message", methods=["POST"])
def handle_event():
    """Handle Lark event subscription"""
    try:
        # Log request details for debugging
        raw_body = request.get_data()
        logger.debug("=== Request Debug ===")
        logger.debug("Headers:")
        for header, value in request.headers.items():
            logger.debug(f"{header}: {value}")
        logger.debug(f"Raw body type: {type(raw_body)}")
        logger.debug(f"Raw body length: {len(raw_body)}")
        logger.debug("=== End Request Debug ===")
        
        # Parse JSON data
        try:
            body = raw_body.decode('utf-8')
            data = json.loads(body)
            logger.debug(f"Parsed request data: {data}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            abort(400, "Invalid JSON data")
            
        # Handle encrypted data first
        if "encrypt" in data:
            logger.debug("Received encrypted data, decrypting...")
            try:
                decrypted_data = decrypt_data(data["encrypt"])
                data = json.loads(decrypted_data)
                logger.debug(f"Decrypted data: {data}")
            except Exception as e:
                logger.error(f"Failed to decrypt data: {e}")
                abort(400, "Failed to decrypt data")
            
        # Handle URL verification challenge - no signature verification needed
        if "challenge" in data:
            logger.info("Received URL verification challenge")
            return jsonify({"challenge": data["challenge"]})
            
        # For all other events, verify headers and signature
        timestamp = request.headers.get("X-Lark-Request-Timestamp")
        nonce = request.headers.get("X-Lark-Request-Nonce")
        signature = request.headers.get("X-Lark-Signature")
        
        if not all([timestamp, nonce, signature]):
            logger.error("Missing required headers for verification")
            abort(401, "Missing required headers")
            
        # Verify signature
        if not verify_signature(timestamp, nonce, raw_body, signature):
            logger.error("Invalid signature")
            abort(401, "Invalid signature")
            
        # Handle encrypted data if present
        if "encrypt" in data:
            logger.debug("Received encrypted data, decrypting...")
            decrypted_data = decrypt_data(data["encrypt"])
            data = json.loads(decrypted_data)
            logger.debug(f"Decrypted data: {data}")
            
        # For v2.0 schema, token is in header.token
        token = data.get("header", {}).get("token") if data.get("schema") == "2.0" else data.get("token")
        if token != VERIFICATION_TOKEN:
            logger.error(f"Invalid verification token. Expected: {VERIFICATION_TOKEN}, Got: {token}")
            abort(401, "Invalid verification token")
            
        # Extract message ID for deduplication
        event = data.get("event", {})
        message = event.get("message", {})
        message_id = message.get("message_id")
        
        # Check if we've already processed this message
        if message_id:
            if message_id in message_cache:
                logger.info(f"Duplicate message detected, skipping: {message_id}")
                return jsonify({"status": "ok"})
            message_cache[message_id] = True

        header = data.get("header", {})
        event_type = header.get("event_type")

        # Only process message events
        if event_type != "im.message.receive_v1":
            return jsonify({"status": "ok"})

        event = data.get("event", {})
        message = event.get("message", {})
        
        # Only process text messages
        if message.get("message_type") != "text":
            return jsonify({"status": "ok"})
            
        content = json.loads(message.get("content", "{}"))
        text = content.get("text", "").strip()
        sender_id = event.get("sender", {}).get("sender_id", {}).get("open_id")

        # Early return if no sender_id
        if not sender_id:
            logger.error("No sender_id found in event")
            return jsonify({"status": "error", "message": "No sender_id found"})

        # Handle command messages
        if text.startswith("/"):
            try:
                handle_command(sender_id, text)
            except Exception as e:
                logger.error(f"Command handling error: {e}")
                send_message(sender_id, f"Error executing command: {str(e)}", "post")
            return jsonify({"status": "ok"})

        # Handle chat messages
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
            
        except Exception as e:
            error_message = str(e).lower()
            logger.error(f"Chat error: {error_message}")
            
            # Map common errors to user-friendly messages
            error_mapping = {
                "credit balance is too low": "Your credit balance is too low.",
                "rate limit": "Rate limit exceeded. Please try again later.",
                "context length": "Input text is too long for this model."
            }
            
            error_text = "An unexpected error occurred."
            for key_phrase, message in error_mapping.items():
                if key_phrase in error_message:
                    error_text = message
                    break
            
            send_message(sender_id, f"Error: {error_text}", "post")

        return jsonify({"status": "ok"})

    except Exception as e:
        logger.error(f"Global error handling event: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 

import boto3
import json

# Define AWS Region
AWS_REGION = "us-east-1"  # Change this based on your AWS Bedrock region

session = boto3.Session(profile_name="inf4bed", region_name=AWS_REGION)

# Initialize Bedrock Runtime Client
bedrock_runtime = session.client("bedrock-runtime")

# Define Model ID (Change based on available models in Bedrock)
#MODEL_ID = "anthropic.claude-v2"  # Example: Anthropic Claude-v2
MODEL_ID = "amazon.nova-pro-v1:0"  # Example: Anthropic Claude-v2

# Define Prompt
prompt_text = "What is the capital of France?"

# Construct the payload based on the model's expected input format
payload = {
    "prompt": prompt_text,
    "max_tokens_to_sample": 200,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
}

# Convert payload to JSON string
payload_json = json.dumps(payload)

# Call AWS Bedrock API
try:
    response = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=payload_json
    )

    # Read and parse response
    response_body = json.loads(response["body"].read().decode("utf-8"))
    print("\nResponse from AWS Bedrock:")
    print(response_body)

except Exception as e:
    print("\nError:", e)

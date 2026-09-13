from ollama import chat

print("Sending image to Qwen...", flush=True)

stream = chat(
    model="qwen3-vl:2b",
    messages=[
        {
            "role": "user",
            "content": (
                "Describe this leaf's colour, spots, and damaged areas. "
                "If you cannot identify a disease confidently, say so."
            ),
            "images": [
                r"D:\Download\leaf-image-procession\leaf-image-procession\archive\plantvillage dataset\color\Apple___healthy\0ce497e6-8184-4ef6-9a9f-c46b66c898c3___RS_HL 6036.JPG"
            ],
        }
    ],
    stream=True,
)

for chunk in stream:
    print(chunk.message.content or "", end="", flush=True)

print("\nFinished.")
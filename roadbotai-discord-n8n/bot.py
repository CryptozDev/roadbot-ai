import atexit
import os
import socket

import aiohttp
import discord
from dotenv import load_dotenv

load_dotenv(".env.bot")
load_dotenv()

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")
COMMAND_PREFIXES = tuple(
    p.strip().lower() for p in os.getenv("BOT_COMMAND_PREFIXES", "!roadbot,!rb").split(",") if p.strip()
)

if not BOT_TOKEN:
    raise RuntimeError("Missing DISCORD_BOT_TOKEN in .env.bot")

if not N8N_WEBHOOK_URL:
    raise RuntimeError("Missing N8N_WEBHOOK_URL in .env.bot")


def acquire_single_instance_lock() -> socket.socket:
    lock_host = "127.0.0.1"
    lock_port = int(os.getenv("BOT_SINGLE_INSTANCE_PORT", "54872"))
    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        lock_socket.bind((lock_host, lock_port))
        lock_socket.listen(1)
    except OSError as error:
        lock_socket.close()
        raise RuntimeError(
            f"RoadBot Relay Bot is already running on {lock_host}:{lock_port}. "
            "Stop the other bot instance before starting a new one."
        ) from error

    atexit.register(lock_socket.close)
    return lock_socket


BOT_INSTANCE_LOCK = acquire_single_instance_lock()

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


async def forward_to_n8n(payload: dict) -> None:
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(N8N_WEBHOOK_URL, json=payload) as response:
            if response.status >= 400:
                text = await response.text()
                raise RuntimeError(f"n8n webhook error {response.status}: {text}")


@client.event
async def on_ready():
    print("=" * 50)
    print("RoadBot Relay Bot started")
    print(f"Logged in as: {client.user}")
    print(f"Webhook URL: {N8N_WEBHOOK_URL}")
    print(f"Commands: {', '.join(COMMAND_PREFIXES)}")
    print("=" * 50)


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    content = (message.content or "").strip()
    lower = content.lower()

    matched_prefix = next((p for p in COMMAND_PREFIXES if lower.startswith(p)), None)
    if not matched_prefix:
        return

    query = content[len(matched_prefix):].strip()
    if not query:
        await message.channel.send("พิมพ์คำถามต่อท้ายคำสั่งด้วย เช่น !roadbot เส้นทางกรุงเทพเชียงใหม่")
        return

    payload = {
        "content": query,
        "raw_content": content,
        "author": {
            "username": message.author.name,
            "id": str(message.author.id),
        },
        "channel_id": str(message.channel.id),
        "guild_id": str(message.guild.id) if message.guild else None,
    }

    try:
        await forward_to_n8n(payload)
        print(f"Forwarded command from {message.author} -> n8n")
    except Exception as error:
        print(f"Failed to forward to n8n: {error}")
        await message.channel.send("ส่งคำขอไปยังระบบไม่สำเร็จ กรุณาลองใหม่อีกครั้ง")


if __name__ == "__main__":
    client.run(BOT_TOKEN)

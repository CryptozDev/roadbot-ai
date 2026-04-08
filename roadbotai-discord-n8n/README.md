# Roadbot AI Ver.n8n Discord

โปรเจคนี้มีแค่ 2 ส่วนที่ใช้งานจริง:

1. `../roadbotai-backend`
   - Backend หลักของ RoadbotAI
   - ใช้ RAG/Model เดิม
   - เวอร์ชันนี้ไม่ต้องใช้ database แล้ว
   - n8n จะเรียก endpoint `POST /api/chat/bot`

2. `./bot.py`
   - Discord bot ที่รับคำสั่งจากผู้ใช้
   - ส่งข้อความต่อเข้า n8n Webhook

## โครงทำงาน

```text
Discord User
  -> bot.py
  -> n8n Webhook
  -> RoadbotAI backend (/api/chat/bot)
  -> n8n ส่งข้อความกลับ Discord Webhook
```

## ไฟล์ที่ต้องใช้

- `bot.py`
- `.env.bot`
- `requirements-bot.txt`
- `../workflow.json`

ไฟล์ backend/node server อื่นในโฟลเดอร์นี้ไม่ใช้แล้ว และถูกเอาออกเพื่อไม่ให้สับสน

## ตั้งค่า bot.py

1. คัดลอก `.env.bot.example` เป็น `.env.bot`
2. ใส่ค่าให้ครบ

```env
DISCORD_BOT_TOKEN=your-discord-bot-token
N8N_WEBHOOK_URL=http://localhost:5678/webhook/roadbotai-n8n-discord
BOT_COMMAND_PREFIXES=!roadbot,!rb
```

3. ติดตั้งแพ็กเกจ Python

```powershell
pip install -r requirements-bot.txt
```

4. รัน bot

```powershell
python bot.py
```

ถ้าใช้ virtual environment ที่โฟลเดอร์หลัก:

```powershell
npm.cmd run bot:venv
```

## ตั้งค่า RoadbotAI backend

ตรวจให้แน่ใจว่า `../roadbotai-backend/.env` มีค่า `BOT_API_TOKEN` และ backend รันอยู่ที่พอร์ต `4000`

backend ตัวนี้ไม่ต้องมี MySQL/Prisma/ตาราง session อีกแล้ว ใช้เฉพาะไฟล์ดัชนีใน `data/` และ Python RAG service

endpoint ที่ n8n จะเรียกคือ:

```text
http://localhost:4000/api/chat/bot
```

## ตั้งค่า n8n

Import workflow นี้:

- `../workflow.json`

แล้วตั้ง environment variables ใน n8n ตามตัวอย่างใน `.env.n8n.example`

```env
ROADBOT_BACKEND_URL=http://localhost:4000/api/chat/bot
ROADBOT_BOT_TOKEN=your-roadbot-backend-bot-token
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx/yyy
```

## วิธีใช้งาน

1. รัน `roadbotai-backend`
2. Activate workflow ใน n8n
3. รัน `bot.py`
4. พิมพ์คำสั่งใน Discord เช่น

```text
!roadbot จากกรุงเทพไปเชียงใหม่เส้นทางไหนเสี่ยงอุบัติเหตุ
```

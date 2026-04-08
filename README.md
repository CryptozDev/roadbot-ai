# 🚧 RoadBot AI – Road Accident Information Chatbot

## 📌 Overview

RoadBot AI คือแชตบอตสำหรับตอบคำถามเกี่ยวกับ **ข้อมูลอุบัติเหตุย้อนหลังบนท้องถนนในประเทศไทย**
ช่วยให้ผู้ใช้ค้นหาข้อมูลได้ง่ายขึ้นผ่านภาษาธรรมชาติ โดยไม่ต้องเปิดดู dataset หรือรายงานด้วยตนเอง

ระบบทำงานในลักษณะ:
**รับคำถาม → วิเคราะห์ประเภทคำถาม → ค้นข้อมูลจาก Dataset/FAISS → สรุปคำตอบ → ส่งกลับผู้ใช้**

---

## 🎯 Problem

- ข้อมูลอุบัติเหตุส่วนใหญ่อยู่ในรูปแบบ dataset และรายงาน จึงเข้าถึงยากสำหรับผู้ใช้ทั่วไป
- ผู้ใช้ต้องใช้เวลาค้นหาข้อมูลจากหลายแหล่งและตีความข้อมูลเอง
- ยังไม่มีระบบที่ช่วยตอบคำถามเกี่ยวกับ **สถิติอุบัติเหตุย้อนหลัง** ได้สะดวกผ่านแชต

---

## 💡 Solution

- พัฒนา AI Chatbot สำหรับตอบคำถามอัตโนมัติเกี่ยวกับอุบัติเหตุทางถนน
- เชื่อมข้อมูลจาก `Google Sheet / local dataset` และระบบค้นคืนข้อมูลแบบ `RAG`
- รองรับการใช้งานผ่าน `Discord Bot`, `Webhook`, และสามารถเชื่อมต่อกับ `n8n` ได้

> หมายเหตุ: เวอร์ชันปัจจุบันเน้น **ข้อมูลย้อนหลัง** ไม่ใช่ข้อมูลจราจรแบบ real-time

---

## ⚙️ Tech Stack

- `Node.js + Express` — Backend API
- `Python + FastAPI` — RAG Service
- `FAISS` — Vector Search
- `Sentence Transformers` — Embedding Model
- `Discord Bot` — Chat Interface
- `n8n` — Workflow Automation / Integration
- `Google Sheet / local dataset` — Data Source

---

## 🧩 System Architecture

![RoadBot Workflow](./assets/workflow-diagram.png)

---

## 🔄 Workflow Concept

1. **Trigger**
   - Discord message
   - Webhook request จาก n8n หรือ Web Chat

2. **Process**
   - รับคำถามจากผู้ใช้
   - Backend API วิเคราะห์ intent ของคำถาม
   - RAG Service ค้นข้อมูลจาก `FAISS index` และ dataset อุบัติเหตุ
   - สรุปและจัดรูปแบบคำตอบเป็นภาษาไทย

3. **Output**
   - ส่งคำตอบกลับไปยังผู้ใช้ผ่าน Discord / Webhook / API

4. **Error Handling**
   - ตรวจสอบสถานะ backend และ RAG service
   - ส่งข้อความ fallback เมื่อระบบยังไม่พร้อมหรือค้นข้อมูลไม่พบ

---

## 📂 Project Structure

```text
roadbot-ai/
├── workflow.json
├── roadbotai-backend/
│   ├── docker-compose.yml
│   ├── package.json
│   ├── data/
│   │   ├── faiss.index
│   │   └── faiss_meta.json
│   ├── rag_service/
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── scripts/
│   │   ├── ingestSheet.js
│   │   └── runRag.mjs
│   └── src/
│       ├── server.js
│       ├── config/
│       ├── routes/
│       ├── services/
│       └── lib/
└── roadbotai-discord-n8n/
    ├── bot.py
    ├── package.json
    ├── requirements-bot.txt
    └── scripts/
        └── runBot.mjs
```

---

## 🚀 Getting Started

### 1. Clone repo

```bash
git clone https://github.com/your-repo/roadbot-ai.git
cd roadbot-ai
```

### 2. Install dependencies

#### Backend
```bash
cd roadbotai-backend
npm install
```

#### Python RAG Service
```bash
python3 -m pip install -r rag_service/requirements.txt
```

#### Discord Bot
```bash
cd ../roadbotai-discord-n8n
python3 -m pip install -r requirements-bot.txt
npm install
```

### 3. Setup environment variables

กำหนดค่าไฟล์:
- `roadbotai-backend/.env`
- `roadbotai-discord-n8n/.env.bot`

ตัวอย่างค่าที่ใช้งาน:

```env
BOT_API_TOKEN=your_token
PY_RAG_URL=http://127.0.0.1:8001
GROQ_API_KEY=your_key
DEFAULT_SHEET_URL=your_google_sheet_url
DEFAULT_SHEET_GID=0
```

### 4. Run project

#### Start backend
```bash
cd roadbotai-backend
npm run dev
```

#### Start Discord bot
```bash
cd ../roadbotai-discord-n8n
python bot.py
```

---

## 🧪 API Testing

สามารถทดสอบได้ผ่าน:

- Discord Bot
- Backend API
- n8n Webhook

ตัวอย่าง health check:

```bash
curl http://127.0.0.1:4000/health
```

---

## 📊 Dataset

ระบบใช้ข้อมูลอุบัติเหตุย้อนหลังจาก dataset ที่นำเข้าเข้าสู่ `FAISS index` เพื่อใช้ค้นคืนข้อมูล

ข้อมูลหลักที่ใช้งาน เช่น:
- จังหวัด
- สายทาง / รหัสสายทาง
- กิโลเมตรที่เกิดเหตุ
- ลักษณะการเกิดเหตุ
- สาเหตุ
- สภาพอากาศ
- จำนวนผู้เสียชีวิต / ผู้บาดเจ็บ
- พิกัด `LATITUDE / LONGITUDE`

---

## 📌 Future Improvements

- เพิ่มการเชื่อมต่อข้อมูลจราจรหรืออุบัติเหตุแบบ real-time
- รองรับ LINE OA / Web Chat เพิ่มเติม
- เพิ่ม dashboard สำหรับสถิติและ visualization
- ปรับปรุง route analysis ให้ละเอียดขึ้นในระดับถนนหรืออำเภอ

---

## 👨‍💻 Author

### RoadBot AI (กลุ่ม Alpha Stack)
- 66053541 จารุกิตติ์ โลบไธสง
- 66073498 ฆนาการ ศรีเพ็ญ
- 66044213 รัตนพล ศรีโนนยาง
- 66080795 นรวัฒน์ ดูการดี
- 66073998 ชำนาญ เกษมสัตย์

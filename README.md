# 🚧 RoadBot AI – Road Accident Information Chatbot

## 📌 Overview

RoadBot AI คือ Chatbot ที่ช่วยให้ผู้ใช้สามารถสอบถามข้อมูลอุบัติเหตุบนท้องถนนได้ง่ายขึ้น
โดยไม่ต้องค้นหาจาก dataset หรือรายงานที่ซับซ้อน

ระบบจะใช้ AI วิเคราะห์คำถาม → ดึงข้อมูลจาก dataset → สรุปคำตอบให้เข้าใจง่าย

---

## 🎯 Problem

* ข้อมูลอุบัติเหตุเข้าถึงยาก (อยู่ใน dataset / report)
* ผู้ใช้ทั่วไปต้องใช้เวลาค้นหานาน
* ไม่มีระบบตอบคำถามแบบ real-time

---

## 💡 Solution

* ใช้ AI Chatbot ตอบคำถามอัตโนมัติ
* เชื่อม dataset (CSV / Database)
* รองรับ Web Chat / LINE / Discord

---

## ⚙️ Tech Stack

* n8n (Workflow Automation)
* OpenAI API
* Node.js
* CSV / Database
* Webhook

---

## 🧩 System Architecture

![RoadBot Workflow](./assets/workflow-diagram.png)

---

## 🔄 Workflow (n8n Concept)

1. Trigger:

   * Webhook / Chat input

2. Process:

   * AI วิเคราะห์คำถาม
   * ตรวจสอบ intent
   * Query dataset หรือเรียก OpenAI

3. Output:

   * ส่งคำตอบกลับ user

4. Error Handling:

   * Log error
   * แจ้ง user เมื่อระบบล้มเหลว

---

## 📂 Project Structure

```
roadbot-ai/
├── docs/              # Workshop documents
├── data/              # Dataset
├── workflows/         # n8n workflow
├── src/               # Core logic
└── tests/             # Testing
```

---

## 🚀 Getting Started

### 1. Clone repo

```
git clone https://github.com/your-repo/roadbot-ai.git
cd roadbot-ai
```

### 2. Install dependencies

```
npm install
```

### 3. Setup ENV

```
cp .env.example .env
```

ใส่ API KEY:

```
OPENAI_API_KEY=your_key
```

### 4. Run project

```
npm run dev
```

---

## 🧪 API Testing

ดูรายละเอียดใน:

```
docs/API_TEST.md
```

---

## 📊 Dataset

* ไฟล์ CSV หรือ Database
* เก็บข้อมูล:

  * สถานที่
  * วันที่
  * จำนวนอุบัติเหตุ
  * ประเภทอุบัติเหตุ

---

## 📌 Future Improvements

* เพิ่ม real-time traffic data
* รองรับ voice input
* เพิ่ม dashboard visualization

---

## 👨‍💻 Author

* RoadBot AI ( กลุ่ม Alpha Stack )
66053541 จารุกิตติ์ โลบไธสง
66073498 ฆนาการ ศรีเพ็ญ
66044213 รัตนพล ศรีโนนยาง
66080795 นรวัฒน์ ดูการดี
66073998 ชำนาญ เกษมสัตย์

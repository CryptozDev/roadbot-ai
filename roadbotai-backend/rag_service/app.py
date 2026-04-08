import heapq
import json
import os
import re
import time
from datetime import datetime
from functools import lru_cache
from math import atan2, cos, radians, sin, sqrt
from typing import Any

import faiss
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI(title="Roadbot RAG Service")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLUMN_MAP = {
    "province": "จังหวัด",
    "road_name": "สายทาง",
    "road_code": "รหัสสายทาง",
    "km": "KM",
    "location": "บริเวณที่เกิดเหตุ",
    "acc_type": "ลักษณะการเกิดเหตุ",
    "cause": "มูลเหตุสันนิษฐาน",
    "weather": "สภาพอากาศ",
    "vehicle_type": "รถคันที่1",
    "dead": "ผู้เสียชีวิต",
    "injured": "รวมจำนวนผู้บาดเจ็บ",
    "injured_ser": "ผู้บาดเจ็บสาหัส",
    "agency": "หน่วยงาน",
    "lat": "LATITUDE",
    "lng": "LONGITUDE",
}

ROADBOT_KEYWORDS = [
    "อุบัติเหตุ",
    "จุดเสี่ยง",
    "เส้นทาง",
    "ระวัง",
    "จังหวัด",
    "สถิติ",
    "ถนน",
    "สายทาง",
    "สาเหตุ",
    "ผู้เสียชีวิต",
    "บาดเจ็บ",
    "สภาพอากาศ",
]
IRRELEVANT_HINTS = ["กินอะไร", "เพลง", "หนัง", "ซีรีส์", "ตลก", "หวย"]
RISK_LOCATION_KEYWORDS = [
    "ทางโค้ง",
    "ทางแยก",
    "จุดกลับรถ",
    "สี่แยก",
    "สามแยก",
    "แยก",
    "ยูเทิร์น",
    "กลับรถ",
    "วงเวียน",
    "คอสะพาน",
    "สะพาน",
    "ทางลอด",
]

THAI_PROVINCES = [
    "กรุงเทพ",
    "กระบี่",
    "กาญจนบุรี",
    "กาฬสินธุ์",
    "กำแพงเพชร",
    "ขอนแก่น",
    "จันทบุรี",
    "ฉะเชิงเทรา",
    "ชลบุรี",
    "ชัยนาท",
    "ชัยภูมิ",
    "ชุมพร",
    "เชียงราย",
    "เชียงใหม่",
    "ตรัง",
    "ตราด",
    "ตาก",
    "นครนายก",
    "นครปฐม",
    "นครพนม",
    "นครราชสีมา",
    "นครศรีธรรมราช",
    "นครสวรรค์",
    "นนทบุรี",
    "นราธิวาส",
    "น่าน",
    "บึงกาฬ",
    "บุรีรัมย์",
    "ปทุมธานี",
    "ประจวบคีรีขันธ์",
    "ปราจีนบุรี",
    "ปัตตานี",
    "พระนครศรีอยุธยา",
    "พะเยา",
    "พัทลุง",
    "พิจิตร",
    "พิษณุโลก",
    "เพชรบุรี",
    "เพชรบูรณ์",
    "แพร่",
    "ภูเก็ต",
    "มหาสารคาม",
    "มุกดาหาร",
    "แม่ฮ่องสอน",
    "ยโสธร",
    "ยะลา",
    "ร้อยเอ็ด",
    "ระนอง",
    "ระยอง",
    "ราชบุรี",
    "ลพบุรี",
    "ลำปาง",
    "ลำพูน",
    "เลย",
    "ศรีสะเกษ",
    "สกลนคร",
    "สงขลา",
    "สตูล",
    "สมุทรปราการ",
    "สมุทรสงคราม",
    "สมุทรสาคร",
    "สระแก้ว",
    "สระบุรี",
    "สิงห์บุรี",
    "สุโขทัย",
    "สุพรรณบุรี",
    "สุราษฎร์ธานี",
    "สุรินทร์",
    "หนองคาย",
    "หนองบัวลำภู",
    "อ่างทอง",
    "อำนาจเจริญ",
    "อุดรธานี",
    "อุตรดิตถ์",
    "อุทัยธานี",
    "อุบลราชธานี",
]

PROVINCE_ALIASES = {
    "กทม": "กรุงเทพ",
    "กทม.": "กรุงเทพ",
    "กรุงเทพฯ": "กรุงเทพ",
    "กรุงเทพมหานคร": "กรุงเทพ",
}

PLACE_ALIASES = {
    "พระราม2": "ถนนพระราม 2, กรุงเทพมหานคร, Thailand",
    "พระราม 2": "ถนนพระราม 2, กรุงเทพมหานคร, Thailand",
    "บางแสน": "บางแสน ชลบุรี, Thailand",
    "พัทยา": "พัทยา ชลบุรี, Thailand",
    "พัทยาเหนือ": "พัทยาเหนือ ชลบุรี, Thailand",
    "พัทยากลาง": "พัทยากลาง ชลบุรี, Thailand",
    "พัทยาใต้": "พัทยาใต้ ชลบุรี, Thailand",
    "แหลมฉบัง": "แหลมฉบัง ชลบุรี, Thailand",
    "มอเตอร์เวย์": "ทางหลวงพิเศษหมายเลข 7 Thailand",
    "บางนา": "บางนา กรุงเทพมหานคร, Thailand",
    "บางปะอิน": "บางปะอิน พระนครศรีอยุธยา, Thailand",
    "โคราช": "นครราชสีมา Thailand",
    "หาดใหญ่": "หาดใหญ่ สงขลา, Thailand",
}

PLACE_PROVINCE_HINTS = {
    "พระราม2": "กรุงเทพ",
    "พระราม 2": "กรุงเทพ",
    "บางนา": "กรุงเทพ",
    "บางแสน": "ชลบุรี",
    "พัทยา": "ชลบุรี",
    "พัทยาเหนือ": "ชลบุรี",
    "พัทยากลาง": "ชลบุรี",
    "พัทยาใต้": "ชลบุรี",
    "แหลมฉบัง": "ชลบุรี",
    "บางปะอิน": "พระนครศรีอยุธยา",
    "โคราช": "นครราชสีมา",
    "หาดใหญ่": "สงขลา",
}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
LONGDO_EVENT_URL = os.getenv("LONGDO_EVENT_URL", "https://event.longdo.com/feed/json")
OPEN_METEO_URL = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")
EXAT_API_BASE_URL = os.getenv("EXAT_API_BASE_URL", "https://exat-man.web.app/api")
OPENAI_WEB_SEARCH_ENABLED = os.getenv("OPENAI_WEB_SEARCH_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
OPENAI_WEB_MODEL = os.getenv("OPENAI_WEB_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
HERE_API_KEY = os.getenv("HERE_API_KEY", "").strip()
HERE_TRAFFIC_URL = os.getenv("HERE_TRAFFIC_URL", "https://data.traffic.hereapi.com/v7/incidents")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()
TOMTOM_TRAFFIC_URL = os.getenv("TOMTOM_TRAFFIC_URL", "https://api.tomtom.com/traffic/services/5/incidentDetails")
EXAT_LOOKBACK_MONTHS = max(1, int(os.getenv("EXAT_LOOKBACK_MONTHS", "3")))
REALTIME_ROUTE_RADIUS_KM = float(os.getenv("REALTIME_ROUTE_RADIUS_KM", "12"))

http = requests.Session()
http.headers.update({"User-Agent": "RoadbotAI/3.0"})

_embedder_cache: dict[str, SentenceTransformer] = {}

_warmup_state = {
    "ready": False,
    "embedder_ready": False,
    "index_ready": False,
    "meta_ready": False,
    "last_error": "",
    "updated_at": 0,
}

_realtime_cache: dict[str, Any] = {
    "fetched_at": 0.0,
    "events": [],
}

_weather_cache: dict[str, Any] = {
    "fetched_at": 0.0,
    "items": {},
}

_state: dict[str, Any] = {
    "rows": [],
    "texts": [],
    "df": pd.DataFrame(),
    "weather_summary": {},
    "llm": None,
    "llm_provider": "none",
    "openai_client": None,
    "index": None,
    "index_path": "",
    "meta_path": "",
    "embedding_model": "",
}


class IngestBody(BaseModel):
    sheet_url: str
    gid: str = "0"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    index_path: str = "./data/faiss.index"
    meta_path: str = "./data/faiss_meta.json"


class ChatBody(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = 8
    qwen_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    index_path: str = "./data/faiss.index"
    meta_path: str = "./data/faiss_meta.json"


def resolve_data_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value

    normalized = os.path.normpath(path_value or "")
    if normalized.startswith(f".{os.sep}"):
        normalized = normalized[2:]

    return os.path.normpath(os.path.join(BASE_DIR, normalized))


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def build_openai_llm(temperature: float) -> Any | None:
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        return None
    openai_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return ChatOpenAI(model=openai_model, temperature=temperature, api_key=openai_key)


def build_groq_llm(temperature: float) -> Any | None:
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        return None
    groq_model = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
    return ChatGroq(model=groq_model, temperature=temperature, api_key=groq_key)


def get_llm(preferred_provider: str | None = None) -> Any | None:
    if _state.get("llm") is not None and (preferred_provider in (None, _state.get("llm_provider"))):
        return _state["llm"]

    temperature = float(os.getenv("LLM_TEMPERATURE", os.getenv("GROQ_TEMPERATURE", "0.1")))
    provider_order = [preferred_provider] if preferred_provider else []
    provider_order.extend(["openai", "groq"])

    tried: set[str] = set()
    for provider in provider_order:
        if not provider or provider in tried:
            continue
        tried.add(provider)
        try:
            llm = build_openai_llm(temperature) if provider == "openai" else build_groq_llm(temperature)
            if llm is not None:
                _state["llm"] = llm
                _state["llm_provider"] = provider
                return llm
        except Exception:
            continue

    _state["llm"] = None
    _state["llm_provider"] = "none"
    return None


def llm_invoke(prompt: str) -> str:
    providers = [_state.get("llm_provider"), "openai", "groq"]
    tried: set[str] = set()

    for provider in providers:
        provider = provider or None
        if provider in tried:
            continue
        tried.add(provider)

        llm = get_llm(provider)
        if llm is None:
            continue

        try:
            return str(llm.invoke(prompt).content or "").strip()
        except Exception:
            _state["llm"] = None
            if provider:
                _state["llm_provider"] = "none"
            continue

    return ""


def get_openai_client() -> OpenAI | None:
    if _state.get("openai_client") is not None:
        return _state["openai_client"]

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)
        _state["openai_client"] = client
        return client
    except Exception:
        _state["openai_client"] = None
        return None


def fetch_openai_web_realtime_summary(
    question: str,
    route_title: str,
    provinces: list[str],
    nearby_events: list[dict[str, Any]],
    current_weather: list[dict[str, Any]],
) -> str:
    if not OPENAI_WEB_SEARCH_ENABLED:
        return ""

    client = get_openai_client()
    if client is None:
        return ""

    event_context = "\n".join(
        [
            f"- [{event.get('source', 'source')}] {event.get('title', '')} | {event.get('description', '')} | {event.get('start', '')}"
            for event in nearby_events[:4]
        ]
    ) or "- ไม่มีเหตุการณ์สดจาก feed ภายในระบบ"

    weather_context = "\n".join(
        [
            f"- {item.get('label', '-')}: {item.get('condition', '-')}, {item.get('temperature_c', '-')}°C, ฝน {item.get('rain_mm', 0)} มม."
            for item in current_weather[:3]
        ]
    ) or "- ไม่มีข้อมูลอากาศล่าสุด"

    prompt = (
        "ค้นข้อมูลเว็บสาธารณะล่าสุดที่เกี่ยวข้องกับการจราจรหรืออุบัติเหตุในประเทศไทยสำหรับเส้นทางนี้ และสรุปแบบสั้น กระชับ เป็นภาษาไทย\n"
        "หากไม่พบข้อมูลใหม่ที่น่าเชื่อถือ ให้บอกว่าไม่พบข้อมูลเว็บเพิ่มเติม\n"
        "ห้ามแต่งข้อมูล และให้เน้นเฉพาะข้อมูลสดหรืออัปเดตล่าสุดเกี่ยวกับการเดินทาง\n\n"
        f"คำถามผู้ใช้: {question}\n"
        f"เส้นทาง: {route_title}\n"
        f"จังหวัดตามแนวเส้นทาง: {' -> '.join(provinces) if provinces else '-'}\n\n"
        f"ข้อมูลสดที่ระบบมีอยู่แล้ว:\n{event_context}\n\n"
        f"สภาพอากาศล่าสุด:\n{weather_context}\n"
    )

    try:
        response = client.responses.create(
            model=OPENAI_WEB_MODEL,
            tools=[
                {
                    "type": "web_search_preview",
                    "user_location": {"type": "approximate", "country": "TH"},
                }
            ],
            input=prompt,
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        return text
    except Exception:
        return ""


def gsheet_to_csv_url(url: str, gid: str) -> str:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Invalid Google Sheet URL")

    sheet_id = match.group(1)
    gid_match = re.search(r"gid=(\d+)", url)
    if gid_match:
        gid = gid_match.group(1)

    gid = (gid or "0").strip() or "0"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def load_sheet_df(sheet_url: str, gid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_url = gsheet_to_csv_url(sheet_url, gid)
    raw = pd.read_csv(csv_url)
    if raw.empty:
        raise ValueError("No rows found in sheet")

    rename_dict = {v: k for k, v in COLUMN_MAP.items() if v in raw.columns}
    df = raw.rename(columns=rename_dict)
    available = [k for k in COLUMN_MAP.keys() if k in df.columns]
    df = df[available].fillna("ไม่ระบุ").astype(str)
    return raw, df


def row_to_text(row: dict[str, str]) -> str:
    parts: list[str] = []
    if row.get("province", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"จังหวัด{row['province']}")
    if row.get("road_name", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"สายทาง{row['road_name']}")
    if row.get("road_code", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"รหัสสายทาง {row['road_code']}")
    if row.get("km", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"กม.ที่ {row['km']}")
    if row.get("location", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"บริเวณ: {row['location']}")
    if row.get("acc_type", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"ลักษณะ: {row['acc_type']}")
    if row.get("cause", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"สาเหตุ: {row['cause']}")
    if row.get("vehicle_type", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"ยานพาหนะ: {row['vehicle_type']}")
    if row.get("weather", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"สภาพอากาศ: {row['weather']}")
    if row.get("dead", "ไม่ระบุ") not in ("ไม่ระบุ", "0", "0.0"):
        parts.append(f"เสียชีวิต {row['dead']} คน")
    if row.get("injured", "ไม่ระบุ") not in ("ไม่ระบุ", "0", "0.0"):
        parts.append(f"บาดเจ็บ {row['injured']} คน")
    if row.get("injured_ser", "ไม่ระบุ") not in ("ไม่ระบุ", "0", "0.0"):
        parts.append(f"บาดเจ็บสาหัส {row['injured_ser']} คน")
    if row.get("agency", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"หน่วยงาน: {row['agency']}")
    return "อุบัติเหตุ: " + " | ".join(parts)


def row_to_metadata(row: dict[str, str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for col in ["province", "road_name", "road_code", "km", "lat", "lng"]:
        if col in row:
            meta[col] = str(row[col])
    return meta


def build_weather_summary(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    if "weather" not in df.columns or "province" not in df.columns:
        return summary

    for province, group in df.groupby("province"):
        weather_counts = group["weather"].value_counts().head(3).to_dict()
        summary[str(province)] = {str(k): int(v) for k, v in weather_counts.items()}
    return summary


def save_meta(meta_path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def load_meta(meta_path: str) -> dict[str, Any] | None:
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        return None
    return payload


def load_state_from_meta(payload: dict[str, Any]) -> None:
    rows = payload.get("rows") or []
    # Support old format (rows have only "raw" with Thai column names) and new format (rows have "std")
    _reverse_map = {v: k for k, v in COLUMN_MAP.items()}

    texts: list[str] = []
    processed_rows: list[dict[str, Any]] = []
    std_rows: list[dict[str, str]] = []

    for item in rows:
        texts.append(str(item.get("content") or ""))
        if "std" in item:
            std = item["std"]
        else:
            # Old format: reconstruct std from raw Thai column names
            raw = item.get("raw") or {}
            std = {_reverse_map.get(k, k): str(v) for k, v in raw.items() if str(v) not in ("", "nan")}
        std_rows.append(std)
        processed_rows.append({**item, "std": std})

    _state["rows"] = processed_rows
    _state["texts"] = texts

    df = pd.DataFrame(std_rows).fillna("ไม่ระบุ").astype(str) if std_rows else pd.DataFrame()
    if "province" in df.columns:
        df["province"] = df["province"].apply(lambda x: normalize_province_name(str(x)) or str(x))
    _state["df"] = df
    _state["weather_summary"] = payload.get("weather_summary") or build_weather_summary(_state["df"])


def prepare_runtime(sheet_url: str, gid: str, embedding_model: str, index_path: str, meta_path: str) -> dict[str, Any]:
    raw_df, df = load_sheet_df(sheet_url, gid)
    rows_payload: list[dict[str, Any]] = []
    texts: list[str] = []

    for row_index, row in df.iterrows():
        std = {k: str(v) for k, v in row.to_dict().items()}
        raw = raw_df.iloc[row_index].fillna("ไม่ระบุ").astype(str).to_dict()
        content = row_to_text(std)
        texts.append(content)
        rows_payload.append(
            {
                "row": int(row_index + 1),
                "content": content,
                "std": std,
                "raw": raw,
                "metadata": row_to_metadata(std),
            }
        )

    embedder = get_embedder(embedding_model)
    vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    vectors = normalize(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)

    weather_summary = build_weather_summary(df)
    payload = {
        "source": {
            "sheet_url": sheet_url,
            "gid": gid,
            "generated_at": int(time.time()),
        },
        "columns": list(df.columns),
        "weather_summary": weather_summary,
        "rows": rows_payload,
    }
    save_meta(meta_path, payload)

    if "province" in df.columns:
        df["province"] = df["province"].apply(lambda x: normalize_province_name(str(x)) or str(x))

    _state["rows"] = rows_payload
    _state["texts"] = texts
    _state["df"] = df
    _state["weather_summary"] = weather_summary
    _state["index"] = index
    _state["index_path"] = index_path
    _state["meta_path"] = meta_path
    _state["embedding_model"] = embedding_model

    return payload


def ensure_runtime_ready(embedding_model: str, index_path: str, meta_path: str) -> bool:
    # in-memory ready
    if _state.get("rows") and _state.get("index") is not None and _state.get("embedding_model") == embedding_model:
        return True

    # on-disk ready
    payload = load_meta(meta_path)
    if payload is not None and os.path.exists(index_path):
        load_state_from_meta(payload)
        _state["index"] = faiss.read_index(index_path)
        _state["index_path"] = index_path
        _state["meta_path"] = meta_path
        _state["embedding_model"] = embedding_model
        return True

    default_sheet_url = os.getenv("DEFAULT_SHEET_URL", "").strip()
    default_gid = (os.getenv("DEFAULT_SHEET_GID", "0") or "0").strip()
    if not default_sheet_url:
        return False

    try:
        prepare_runtime(default_sheet_url, default_gid, embedding_model, index_path, meta_path)
        return True
    except Exception:
        return False


def normalize_province_name(name: str | None) -> str | None:
    if not name:
        return None

    cleaned = str(name).strip()
    cleaned = cleaned.replace("จังหวัด", "").replace("จ.", "").replace("ฯ", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = PROVINCE_ALIASES.get(cleaned, cleaned)

    for province in THAI_PROVINCES:
        if cleaned == province or cleaned in province or province in cleaned:
            return province
    return cleaned or None


def combine_ordered_provinces(*items: Any) -> list[str]:
    ordered: list[str] = []

    def add_one(value: Any) -> None:
        normalized = normalize_province_name(str(value) if value is not None else "")
        if normalized and normalized not in ordered:
            ordered.append(normalized)

    for item in items:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            for value in item:
                add_one(value)
        else:
            add_one(item)

    return ordered


def extract_provinces_from_text(text: str) -> list[str]:
    text = str(text or "")
    matches: list[tuple[int, str]] = []
    candidates = list(
        dict.fromkeys(sorted(THAI_PROVINCES + list(PROVINCE_ALIASES.keys()), key=len, reverse=True))
    )

    for name in candidates:
        start = 0
        while True:
            idx = text.find(name, start)
            if idx == -1:
                break
            normalized = normalize_province_name(name)
            if normalized:
                matches.append((idx, normalized))
            start = idx + len(name)

    matches.sort(key=lambda item: item[0])
    ordered: list[str] = []
    for _, province in matches:
        if province not in ordered:
            ordered.append(province)
    return ordered


def has_risk_keyword(text: str) -> bool:
    content = str(text or "")
    return any(keyword in content for keyword in RISK_LOCATION_KEYWORDS)


def extract_origin_destination(question: str) -> tuple[str | None, str | None, list[str]]:
    question = str(question or "").strip()
    waypoints: list[str] = []

    for keyword in ["ผ่าน", "ไปทาง", "แวะ", "via"]:
        if keyword in question:
            tail = question.split(keyword, 1)[1]
            waypoints = extract_provinces_from_text(tail)
            break

    if "จาก" in question and "ไป" in question:
        from_idx = question.find("จาก")
        to_idx = question.find("ไป")
        if from_idx < to_idx:
            origin_text = question[from_idx + len("จาก") : to_idx]
            destination_text = question[to_idx + len("ไป") :]
        else:
            destination_text = question[to_idx + len("ไป") : from_idx]
            origin_text = question[from_idx + len("จาก") :]

        origin_candidates = extract_provinces_from_text(origin_text)
        destination_candidates = extract_provinces_from_text(destination_text)
        origin = origin_candidates[0] if origin_candidates else normalize_province_name(origin_text)
        destination = (
            destination_candidates[0] if destination_candidates else normalize_province_name(destination_text)
        )
        waypoints = [point for point in waypoints if point not in {origin, destination}]

        if origin and destination:
            return origin, destination, waypoints

    route_match = re.search(
        r"([ก-๙A-Za-z\.\sฯ]+?)\s*ไป\s*([ก-๙A-Za-z\.\sฯ]+?)(?:\s*(?:ผ่าน|ไปทาง|แวะ|มี|ควร|$))",
        question,
    )
    if route_match:
        origin_candidates = extract_provinces_from_text(route_match.group(1))
        destination_candidates = extract_provinces_from_text(route_match.group(2))
        origin = origin_candidates[0] if origin_candidates else normalize_province_name(route_match.group(1))
        destination = (
            destination_candidates[0] if destination_candidates else normalize_province_name(route_match.group(2))
        )
        waypoints = [point for point in waypoints if point not in {origin, destination}]

        if origin and destination:
            return origin, destination, waypoints

    detected = extract_provinces_from_text(question)
    if len(detected) >= 2:
        if question.startswith("ไป") and "จาก" not in question:
            return detected[-1], detected[0], detected[1:-1]
        return detected[0], detected[-1], detected[1:-1]

    prompt = (
        "จากคำถามต่อไปนี้ ให้ดึงชื่อ ต้นทาง จังหวัดที่แวะผ่าน (ถ้ามี) และ ปลายทาง\n"
        "ตอบในรูปแบบ: ORIGIN|WAYPOINT1,WAYPOINT2|DESTINATION\n"
        "ถ้าไม่มี waypoint ให้ใส่ว่าง: ORIGIN||DESTINATION\n"
        "ใช้ชื่อจังหวัดภาษาไทยเท่านั้น\n"
        "ถ้าไม่มีต้นทางหรือปลายทาง ตอบ: NONE||NONE\n\n"
        f"คำถาม: {question}\nตอบ:"
    )
    response = llm_invoke(prompt)
    first_line = response.split("\n")[0].strip()
    parts = first_line.split("|")

    if len(parts) == 3:
        origin = normalize_province_name(parts[0].strip())
        destination = normalize_province_name(parts[2].strip())
        waypoints = [
            normalize_province_name(item.strip())
            for item in parts[1].split(",")
            if item.strip() and item.strip() != "NONE"
        ]
        waypoints = [item for item in waypoints if item and item not in {origin, destination}]
        if origin and destination:
            return origin, destination, waypoints

    return None, None, []


def is_known_province(value: str | None) -> bool:
    normalized = normalize_province_name(value)
    return bool(normalized and normalized in THAI_PROVINCES)


def clean_place_label(text: str | None) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"[`\"'“”‘’]+", "", cleaned)
    cleaned = re.split(
        r"(?:ตอนนี้|ล่าสุด|เรียลไทม์|สด|คืนนี้|พรุ่งนี้|วันนี้|เช้านี้|เย็นนี้|ช่วงนี้|ดึกนี้|จุดเสี่ยง|อุบัติเหตุ|รถติด|ต้องระวัง|ควรระวัง|ควรเลี่ยง|เส้นทางไหนบ้าง|เส้นทางไหน|จากข้อมูล|ข้อมูลสด|ข้อมูลย้อนหลัง|มีอะไร|มี|ไหม|มั้ย|หรือไม่|ผ่าน|แวะ|via)",
        cleaned,
        maxsplit=1,
    )[0]
    cleaned = re.split(r"\s+(?:จาก|ไป|ถึง)\s+", cleaned, maxsplit=1)[0]
    cleaned = cleaned.strip(" ,-/|>~")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def resolve_place_to_province(place_name: str | None) -> str | None:
    label = clean_place_label(place_name)
    if not label:
        return None

    hinted = PLACE_PROVINCE_HINTS.get(label)
    if hinted:
        return hinted

    found = extract_provinces_from_text(label)
    if found:
        return found[0]

    normalized = normalize_province_name(label)
    if normalized in THAI_PROVINCES:
        return normalized

    lat, lng = geocode(label)
    if lat is not None and lng is not None:
        province = reverse_geocode_province(lat, lng)
        if province:
            return province

    return normalized if normalized in THAI_PROVINCES else None


def extract_route_labels(question: str) -> tuple[str | None, str | None, list[str]]:
    text = str(question or "").strip()
    if not text:
        return None, None, []

    origin_match = re.search(
        r"จาก\s*([ก-๙A-Za-z0-9\.\-/\s]+?)(?=$|ผ่าน|แวะ|via|ไป|ถึง|คืนนี้|พรุ่งนี้|วันนี้|เช้านี้|เย็นนี้|ช่วงนี้|จุดเสี่ยง|อุบัติเหตุ|รถติด|ไหม|มั้ย|ล่าสุด|ตอนนี้|มี|ต้อง|ควร|ระวัง)",
        text,
    )
    destination_match = re.search(
        r"(?:ไป|ถึง)\s*([ก-๙A-Za-z0-9\.\-/\s]+?)(?=$|จาก|ผ่าน|แวะ|via|คืนนี้|พรุ่งนี้|วันนี้|เช้านี้|เย็นนี้|ช่วงนี้|จุดเสี่ยง|อุบัติเหตุ|รถติด|ไหม|มั้ย|ล่าสุด|ตอนนี้|มี|ต้อง|ควร|ระวัง)",
        text,
    )
    waypoint_matches = re.findall(
        r"(?:ผ่าน|แวะ|via)\s*([ก-๙A-Za-z0-9\.\-/\s]+?)(?=$|จาก|ผ่าน|แวะ|via|ไป|ถึง|คืนนี้|พรุ่งนี้|วันนี้|เช้านี้|เย็นนี้|ช่วงนี้|จุดเสี่ยง|อุบัติเหตุ|รถติด|ไหม|มั้ย|ล่าสุด|ตอนนี้|มี|ต้อง|ควร|ระวัง)",
        text,
    )

    origin_label = clean_place_label(origin_match.group(1)) if origin_match else None
    destination_label = clean_place_label(destination_match.group(1)) if destination_match else None
    waypoint_labels = [clean_place_label(item) for item in waypoint_matches if clean_place_label(item)]

    named_candidates = [label for label in sorted(PLACE_ALIASES.keys(), key=len, reverse=True) if label in text]
    province_candidates = extract_provinces_from_text(text)
    combined_candidates: list[str] = []
    for item in named_candidates + province_candidates:
        cleaned = clean_place_label(item)
        if cleaned and cleaned not in combined_candidates:
            combined_candidates.append(cleaned)

    if len(combined_candidates) >= 2:
        reversed_route_order = text.startswith("ไป") and "จาก" in text and text.find("ไป") < text.find("จาก")
        if not origin_label:
            origin_label = combined_candidates[-1] if reversed_route_order else combined_candidates[0]
        if not destination_label:
            destination_label = combined_candidates[0] if reversed_route_order else combined_candidates[-1]
    if not waypoint_labels and len(combined_candidates) >= 3:
        waypoint_labels = combined_candidates[1:-1]

    waypoint_labels = [item for item in waypoint_labels if item and item not in {origin_label, destination_label}]
    return origin_label, destination_label, waypoint_labels


def build_route_context(question: str) -> dict[str, Any]:
    origin_label, destination_label, waypoint_labels = extract_route_labels(question)
    fallback_origin, fallback_destination, fallback_waypoints = extract_origin_destination(question)

    if not origin_label and fallback_origin:
        origin_label = clean_place_label(fallback_origin)
    if not destination_label and fallback_destination:
        destination_label = clean_place_label(fallback_destination)
    if not waypoint_labels and fallback_waypoints:
        waypoint_labels = [clean_place_label(item) for item in fallback_waypoints if clean_place_label(item)]

    place_labels: list[str] = []
    for item in [origin_label, *waypoint_labels, destination_label]:
        cleaned = clean_place_label(item)
        if cleaned and cleaned not in place_labels:
            place_labels.append(cleaned)

    origin = resolve_place_to_province(origin_label or fallback_origin)
    destination = resolve_place_to_province(destination_label or fallback_destination)

    waypoint_provinces: list[str] = []
    for item in waypoint_labels or fallback_waypoints:
        province = resolve_place_to_province(item)
        if province and province not in waypoint_provinces and province not in {origin, destination}:
            waypoint_provinces.append(province)

    return {
        "origin": origin,
        "destination": destination,
        "waypoints": waypoint_provinces,
        "origin_label": origin_label or origin,
        "destination_label": destination_label or destination,
        "waypoint_labels": waypoint_labels,
        "place_labels": place_labels,
    }


def classify_question(question: str) -> str:
    question = str(question or "").strip()
    found_provinces = extract_provinces_from_text(question)
    origin_label, destination_label, waypoint_labels = extract_route_labels(question)
    has_route_markers = any(word in question for word in ["ไป", "จาก", "ผ่าน", "เส้นทาง", "จุดเสี่ยง", "ระวัง", "ขับ", "แวะ", "via"])
    has_route_context = bool(origin_label and destination_label) or bool(destination_label and waypoint_labels)

    if any(hint in question for hint in IRRELEVANT_HINTS):
        return "irrelevant"

    has_realtime_signal = any(word in question for word in ["ตอนนี้", "ล่าสุด", "เรียลไทม์", "สด", "คืนนี้", "วันนี้", "พรุ่งนี้", "ช่วงนี้", "live", "now"])

    if has_realtime_signal and (has_route_context or len(found_provinces) >= 1):
        return "realtime"

    if has_route_markers and (has_route_context or len(found_provinces) >= 2):
        return "route_analysis"

    if has_realtime_signal:
        return "realtime"

    if any(
        word in question
        for word in ["จังหวัดที่มีอุบัติเหตุ", "จังหวัดไหนมีอุบัติเหตุ", "มากสุด", "สูงสุด", "มากที่สุด"]
    ):
        return "top_provinces"

    if any(word in question for word in ["สาเหตุ", "ต้นเหตุ", "เกิดจากอะไร"]):
        return "top_causes"

    if found_provinces and any(word in question for word in ["อุบัติเหตุ", "จุดเสี่ยง", "จังหวัด", "สภาพอากาศ", "ระวัง"]):
        return "province_details"

    if any(word in question for word in ROADBOT_KEYWORDS):
        return "other_accident_stats"

    prompt = (
        "จากคำถามต่อไปนี้ ให้จัดหมวดหมู่เป็นประเภทใดประเภทหนึ่งเท่านั้น:\n"
        "- route_analysis\n- top_provinces\n- top_causes\n"
        "- province_details\n- realtime\n- irrelevant\n- other_accident_stats\n\n"
        f"คำถาม: {question}\nหมวดหมู่:"
    )
    response = llm_invoke(prompt)
    valid = {
        "route_analysis",
        "top_provinces",
        "top_causes",
        "province_details",
        "realtime",
        "irrelevant",
        "other_accident_stats",
    }
    return response if response in valid else "irrelevant"


def get_weather_summary(provinces: list[str]) -> str:
    lines: list[str] = []
    weather_summary = _state.get("weather_summary") or {}
    for province in provinces:
        if province in weather_summary:
            counts = weather_summary[province]
            top = ", ".join([f"{weather} ({count} ครั้ง)" for weather, count in counts.items()])
            lines.append(f"- {province}: อุบัติเหตุเกิดบ่อยในสภาพ {top}")
    return "\n".join(lines) if lines else "ไม่มีข้อมูลสภาพอากาศ"


def build_province_stats_summary(province: str) -> str:
    df = _state.get("df")
    province = normalize_province_name(province)
    if province is None or not isinstance(df, pd.DataFrame) or df.empty or "province" not in df.columns:
        return "ไม่มีข้อมูลสถิติเพิ่มเติม"

    province_df = df[df["province"].astype(str) == province].copy()
    if province_df.empty:
        return f"- {province}: ไม่มีข้อมูลในตาราง"

    lines = [f"- {province}: พบอุบัติเหตุ {len(province_df):,} รายการ"]

    if "cause" in province_df.columns:
        top_cause = province_df[province_df["cause"] != "ไม่ระบุ"]["cause"].value_counts().head(3)
        if not top_cause.empty:
            lines.append(
                "  สาเหตุหลัก: "
                + ", ".join([f"{cause} ({count} ครั้ง)" for cause, count in top_cause.items()])
            )

    if "location" in province_df.columns:
        top_locations = province_df[province_df["location"] != "ไม่ระบุ"]["location"].value_counts().head(3)
        if not top_locations.empty:
            lines.append(
                "  จุดที่พบบ่อย: "
                + ", ".join([f"{location} ({count} ครั้ง)" for location, count in top_locations.items()])
            )

    if "weather" in province_df.columns:
        top_weather = province_df[province_df["weather"] != "ไม่ระบุ"]["weather"].value_counts().head(2)
        if not top_weather.empty:
            lines.append(
                "  สภาพอากาศที่พบมาก: "
                + ", ".join([f"{weather} ({count} ครั้ง)" for weather, count in top_weather.items()])
            )

    if "dead" in province_df.columns:
        dead_total = pd.to_numeric(province_df["dead"], errors="coerce").fillna(0).sum()
        if dead_total > 0:
            lines.append(f"  ผู้เสียชีวิตสะสม: {int(dead_total):,} คน")

    return "\n".join(lines)


def build_overall_stats_summary() -> str:
    df = _state.get("df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "ไม่มีข้อมูลภาพรวม"

    lines = [f"- จำนวนข้อมูลอุบัติเหตุทั้งหมด: {len(df):,} รายการ"]

    if "province" in df.columns:
        top_provinces = df[df["province"] != "ไม่ระบุ"]["province"].value_counts().head(5)
        if not top_provinces.empty:
            lines.append(
                "- จังหวัดที่พบมากสุด: "
                + ", ".join([f"{province} ({count} ครั้ง)" for province, count in top_provinces.items()])
            )

    if "cause" in df.columns:
        top_causes = df[df["cause"] != "ไม่ระบุ"]["cause"].value_counts().head(5)
        if not top_causes.empty:
            lines.append(
                "- สาเหตุที่พบบ่อย: "
                + ", ".join([f"{cause} ({count} ครั้ง)" for cause, count in top_causes.items()])
            )

    if "weather" in df.columns:
        top_weather = df[df["weather"] != "ไม่ระบุ"]["weather"].value_counts().head(3)
        if not top_weather.empty:
            lines.append(
                "- สภาพอากาศที่พบบ่อย: "
                + ", ".join([f"{weather} ({count} ครั้ง)" for weather, count in top_weather.items()])
            )

    return "\n".join(lines)


def extract_route_risk_points(provinces: list[str], limit_per_province: int = 2, total_limit: int = 10) -> list[dict]:
    df = _state.get("df")
    if not isinstance(df, pd.DataFrame) or df.empty or "province" not in df.columns:
        return []

    results: list[dict] = []
    for province in provinces:
        province_df = df[df["province"].astype(str) == province].copy()
        if province_df.empty:
            continue

        province_df["_dead_num"] = pd.to_numeric(province_df.get("dead", 0), errors="coerce").fillna(0)
        province_df["_injured_num"] = pd.to_numeric(province_df.get("injured", 0), errors="coerce").fillna(0)
        province_df["_injured_ser_num"] = pd.to_numeric(province_df.get("injured_ser", 0), errors="coerce").fillna(0)
        province_df["_severity"] = (
            province_df["_dead_num"] * 5 + province_df["_injured_ser_num"] * 3 + province_df["_injured_num"]
        )

        location_series = (
            province_df["location"].astype(str)
            if "location" in province_df.columns
            else pd.Series("", index=province_df.index)
        )
        acc_series = (
            province_df["acc_type"].astype(str)
            if "acc_type" in province_df.columns
            else pd.Series("", index=province_df.index)
        )
        risk_mask = location_series.apply(has_risk_keyword) | acc_series.apply(has_risk_keyword)
        risk_df = province_df[risk_mask].copy()

        if risk_df.empty:
            if "location" in province_df.columns:
                risk_df = province_df[province_df["location"].astype(str) != "ไม่ระบุ"].copy()
            else:
                risk_df = province_df.copy()
        if risk_df.empty:
            risk_df = province_df.copy()

        risk_df = risk_df.sort_values(["_severity"], ascending=False).head(limit_per_province)
        for _, row in risk_df.iterrows():
            results.append(row.to_dict())

    results = sorted(results, key=lambda item: float(item.get("_severity", 0)), reverse=True)
    return results[:total_limit]


def build_risk_point_line(row: dict[str, Any]) -> str:
    parts: list[str] = []
    if row.get("province", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"จังหวัด{row['province']}")
    if row.get("road_name", "ไม่ระบุ") != "ไม่ระบุ":
        road_text = f"สายทาง {row['road_name']}"
        if row.get("road_code", "ไม่ระบุ") != "ไม่ระบุ":
            road_text += f" ({row['road_code']})"
        parts.append(road_text)
    if row.get("km", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"กม. {row['km']}")
    if row.get("location", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"บริเวณ {row['location']}")
    if row.get("cause", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"สาเหตุ {row['cause']}")
    if row.get("weather", "ไม่ระบุ") != "ไม่ระบุ":
        parts.append(f"อากาศ {row['weather']}")
    if row.get("dead", "ไม่ระบุ") not in ("ไม่ระบุ", "0", "0.0"):
        parts.append(f"เสียชีวิต {row['dead']} คน")
    if row.get("injured", "ไม่ระบุ") not in ("ไม่ระบุ", "0", "0.0"):
        parts.append(f"บาดเจ็บ {row['injured']} คน")
    return "- " + " | ".join(parts)


def infer_route_provinces_with_llm(
    origin: str,
    destination: str,
    waypoints: list[str] | None = None,
    include_waypoints: bool = True,
    force_include_waypoints: bool = False,
) -> list[str]:
    waypoints = waypoints or []
    origin = normalize_province_name(origin)
    destination = normalize_province_name(destination)
    normalized_waypoints = [normalize_province_name(item) for item in waypoints if normalize_province_name(item)]

    if not origin or not destination:
        return combine_ordered_provinces(origin, normalized_waypoints, destination)

    waypoint_text = f" ผ่าน {' -> '.join(normalized_waypoints)}" if include_waypoints and normalized_waypoints else ""
    prompt = (
        "ช่วยคาดการณ์จังหวัดที่เส้นทางรถยนต์ในประเทศไทยน่าจะขับผ่าน โดยเรียงจากต้นทางไปปลายทาง\n"
        f"เส้นทาง: {origin} -> {destination}{waypoint_text}\n\n"
        "กติกา:\n"
        "- ตอบเฉพาะชื่อจังหวัดภาษาไทยเรียงลำดับ\n"
        "- คั่นด้วยเครื่องหมาย | เท่านั้น\n"
        "- ใช้เฉพาะจังหวัดที่มีจริงในไทย\n"
        "- หาก waypoint ดูอ้อมเกินจริง ให้ละทิ้ง waypoint นั้น\n"
        "- หากไม่แน่ใจ ให้ใส่อย่างน้อยต้นทางและปลายทาง\n"
    )
    response = llm_invoke(prompt)

    inferred = [
        normalize_province_name(item.strip())
        for item in re.split(r"\||,|→|->|\n", response)
        if item.strip()
    ]
    if force_include_waypoints and include_waypoints:
        return combine_ordered_provinces(origin, inferred, normalized_waypoints, destination)
    return combine_ordered_provinces(origin, inferred, destination)


@lru_cache(maxsize=256)
def geocode(place_name: str) -> tuple[float | None, float | None]:
    place_name = clean_place_label(place_name)
    if not place_name:
        return None, None

    alias_query = PLACE_ALIASES.get(place_name)
    normalized = normalize_province_name(place_name) or place_name
    queries = [query for query in [alias_query, f"{normalized}, Thailand", f"จังหวัด{normalized}, Thailand", str(place_name)] if query]

    for query in queries:
        try:
            response = http.get(
                NOMINATIM_URL,
                params={
                    "q": query,
                    "format": "jsonv2",
                    "limit": 1,
                    "countrycodes": "th",
                },
                timeout=20,
            )
            data = response.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
        except Exception:
            continue

    return None, None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return earth_radius * c


def detour_metrics(origin: str, waypoint: str, destination: str) -> tuple[float | None, float | None, float | None]:
    orig_lat, orig_lng = get_province_position(origin)
    wp_lat, wp_lng = get_province_position(waypoint)
    dest_lat, dest_lng = get_province_position(destination)

    if None in (orig_lat, orig_lng, wp_lat, wp_lng, dest_lat, dest_lng):
        return None, None, None

    direct = haversine_km(orig_lat, orig_lng, dest_lat, dest_lng)
    via = haversine_km(orig_lat, orig_lng, wp_lat, wp_lng) + haversine_km(wp_lat, wp_lng, dest_lat, dest_lng)
    extra = via - direct
    return direct, via, extra


def filter_unreasonable_waypoints(
    origin: str,
    destination: str,
    waypoints: list[str] | None = None,
    return_removed: bool = False,
) -> tuple[list[str], list[str]] | list[str]:
    waypoints = combine_ordered_provinces(waypoints or [])
    if not waypoints:
        return ([], []) if return_removed else []

    base_route = estimate_route_from_centroids(origin, destination, [])
    if len(base_route) < 3:
        base_route = infer_route_provinces_with_llm(origin, destination, [], include_waypoints=False)

    filtered: list[str] = []
    removed_messages: list[str] = []
    current_origin = origin

    for waypoint in waypoints:
        keep = True
        reason = ""
        direct_km, via_km, extra_km = detour_metrics(current_origin, waypoint, destination)
        route_with_waypoint = estimate_route_from_centroids(current_origin, destination, [waypoint])
        if len(route_with_waypoint) < 3:
            route_with_waypoint = infer_route_provinces_with_llm(
                current_origin,
                destination,
                [waypoint],
                include_waypoints=True,
                force_include_waypoints=False,
            )

        route_signal_reliable = len(base_route) >= 3 and len(route_with_waypoint) >= 3
        route_misses_waypoint = (
            route_signal_reliable
            and waypoint not in base_route
            and waypoint not in route_with_waypoint
        )

        if direct_km is not None and via_km is not None and extra_km is not None:
            ratio = via_km / direct_km if direct_km > 0 else 999
            if via_km > direct_km * 1.45 and extra_km > 180:
                keep = False
                reason = f"ทำให้เส้นทางอ้อมเพิ่มประมาณ {extra_km:.0f} กม. ({ratio:.2f} เท่าของเส้นทางตรง)"
            elif route_misses_waypoint and (via_km > direct_km * 1.20 or extra_km > 120):
                keep = False
                reason = "ไม่อยู่ในแนวเส้นทางหลักที่ระบบคำนวณใหม่ได้"
        else:
            if route_misses_waypoint:
                keep = False
                reason = "ไม่อยู่ในแนวเส้นทางหลักที่ระบบคำนวณใหม่ได้"

        if keep:
            filtered.append(waypoint)
            current_origin = waypoint
        else:
            removed_messages.append(f"- {waypoint}: {reason or 'ไม่สมเหตุสมผลกับเส้นทาง'}")

    if return_removed:
        return filtered, removed_messages
    return filtered


@lru_cache(maxsize=2048)
def reverse_geocode_province(lat: float, lng: float) -> str | None:
    try:
        response = http.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lng, "format": "jsonv2"},
            timeout=5,
        )
        if response.status_code != 200:
            return None
        addr = response.json().get("address", {})
        province_raw = addr.get("province") or addr.get("state") or ""
        return normalize_province_name(str(province_raw))
    except Exception:
        return None


def get_province_centroids() -> dict[str, tuple[float, float]]:
    df = _state.get("df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    if not {"province", "lat", "lng"}.issubset(df.columns):
        return {}

    points = df[["province", "lat", "lng"]].copy()
    points["lat"] = pd.to_numeric(points["lat"], errors="coerce")
    points["lng"] = pd.to_numeric(points["lng"], errors="coerce")
    points = points.dropna(subset=["province", "lat", "lng"])

    centroids: dict[str, tuple[float, float]] = {}
    for province, group in points.groupby("province"):
        normalized = normalize_province_name(str(province))
        if normalized:
            centroids[normalized] = (float(group["lat"].mean()), float(group["lng"].mean()))
    return centroids


def get_province_position(province: str) -> tuple[float | None, float | None]:
    normalized = normalize_province_name(province) or province
    centroids = get_province_centroids()
    if normalized in centroids:
        return centroids[normalized]
    return geocode(normalized)


def estimate_distance_from_route(provinces: list[str] | None = None) -> float:
    route = combine_ordered_provinces(provinces or [])
    if len(route) < 2:
        return 0.0

    total_km = 0.0
    segments = 0
    for start, end in zip(route, route[1:]):
        start_lat, start_lng = get_province_position(start)
        end_lat, end_lng = get_province_position(end)
        if None in (start_lat, start_lng, end_lat, end_lng):
            continue
        total_km += haversine_km(start_lat, start_lng, end_lat, end_lng)
        segments += 1

    if segments == 0:
        return 0.0

    road_factor = 1.12 if segments >= 3 else 1.18
    return total_km * road_factor


def estimate_leg_provinces_from_centroids(origin: str, destination: str) -> list[str]:
    origin = normalize_province_name(origin) or origin
    destination = normalize_province_name(destination) or destination
    centroids = get_province_centroids()

    if origin not in centroids or destination not in centroids:
        orig_lat, orig_lng = get_province_position(origin)
        dest_lat, dest_lng = get_province_position(destination)
        if None in (orig_lat, orig_lng, dest_lat, dest_lng):
            return combine_ordered_provinces(origin, destination)
        return combine_ordered_provinces(origin, destination)

    def province_distance(first: str, second: str) -> float:
        first_lat, first_lng = centroids[first]
        second_lat, second_lng = centroids[second]
        return haversine_km(first_lat, first_lng, second_lat, second_lng)

    direct_km = province_distance(origin, destination)
    if direct_km <= 1:
        return combine_ordered_provinces(origin, destination)

    max_edge_km = 150.0 if direct_km < 260 else 170.0
    neighbor_limit = 12
    progress_slack_km = 30.0
    graph: dict[str, list[tuple[str, float]]] = {}

    for province in centroids:
        candidates: list[tuple[float, str]] = []
        current_goal_km = province_distance(province, destination)
        for other in centroids:
            if other == province:
                continue
            edge_km = province_distance(province, other)
            if edge_km <= max_edge_km and province_distance(other, destination) <= current_goal_km + progress_slack_km:
                candidates.append((edge_km, other))
        candidates.sort(key=lambda item: item[0])
        graph[province] = [(other, edge_km) for edge_km, other in candidates[:neighbor_limit]]

    queue: list[tuple[float, str]] = [(province_distance(origin, destination), origin)]
    costs: dict[str, float] = {origin: 0.0}
    previous: dict[str, str] = {}
    visited: set[str] = set()

    while queue:
        _, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)

        if current == destination:
            break

        for neighbor, edge_km in graph.get(current, []):
            step_cost = (max(edge_km, 1.0) ** 1.4) / (100.0 ** 0.4)
            new_cost = costs[current] + step_cost
            if new_cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = new_cost
                previous[neighbor] = current
                priority = new_cost + (province_distance(neighbor, destination) / 100.0)
                heapq.heappush(queue, (priority, neighbor))

    if destination == origin:
        return [origin]
    if destination not in previous:
        return combine_ordered_provinces(origin, destination)

    path = [destination]
    while path[-1] != origin:
        prior = previous.get(path[-1])
        if prior is None:
            return combine_ordered_provinces(origin, destination)
        path.append(prior)

    path.reverse()
    return combine_ordered_provinces(path)


def estimate_route_from_centroids(origin: str, destination: str, waypoints: list[str] | None = None) -> list[str]:
    stops = combine_ordered_provinces(origin, waypoints or [], destination)
    estimated: list[str] = []
    for leg_origin, leg_destination in zip(stops, stops[1:]):
        estimated = combine_ordered_provinces(estimated, estimate_leg_provinces_from_centroids(leg_origin, leg_destination))
    return estimated


def build_sensible_route(
    origin: str,
    destination: str,
    waypoints: list[str] | None = None,
    observed_provinces: list[str] | None = None,
) -> list[str]:
    waypoints = combine_ordered_provinces(waypoints or [])
    observed_provinces = combine_ordered_provinces(observed_provinces or [])
    stops = combine_ordered_provinces(origin, waypoints, destination)

    if observed_provinces:
        observed_route = combine_ordered_provinces(origin, observed_provinces, destination)
        if len(observed_route) > len(stops) and all(point in observed_route for point in waypoints):
            return observed_route

    estimated_route = estimate_route_from_centroids(origin, destination, waypoints)
    if len(estimated_route) > len(stops):
        return estimated_route

    backbone: list[str] = []
    for leg_origin, leg_destination in zip(stops, stops[1:]):
        inferred_leg = infer_route_provinces_with_llm(
            leg_origin,
            leg_destination,
            [],
            include_waypoints=False,
            force_include_waypoints=False,
        )
        backbone = combine_ordered_provinces(backbone, inferred_leg or [leg_origin, leg_destination])

    if not backbone:
        backbone = stops

    if not observed_provinces:
        return backbone

    observed_route = combine_ordered_provinces(origin, observed_provinces, destination)
    overlap = [province for province in observed_route if province in backbone]
    if len(observed_route) >= len(backbone) or len(overlap) >= max(2, len(backbone) // 2):
        return observed_route

    return backbone


@lru_cache(maxsize=512)
def get_route_leg_provinces(origin: str, destination: str) -> tuple[list[str], float]:
    origin = normalize_province_name(origin) or origin
    destination = normalize_province_name(destination) or destination
    fallback = combine_ordered_provinces(origin, destination)
    estimated_route = estimate_leg_provinces_from_centroids(origin, destination)
    estimated_distance_km = estimate_distance_from_route(estimated_route or fallback)

    orig_lat, orig_lng = get_province_position(origin)
    dest_lat, dest_lng = get_province_position(destination)
    if orig_lat is None or dest_lat is None:
        return estimated_route or fallback, estimated_distance_km

    try:
        response = http.get(
            f"{OSRM_URL}/{orig_lng},{orig_lat};{dest_lng},{dest_lat}",
            params={"overview": "false"},
            timeout=3,
        )
        data = response.json()
    except Exception:
        return estimated_route or fallback, estimated_distance_km

    if data.get("code") != "Ok":
        return estimated_route or fallback, estimated_distance_km

    distance_km = float(data["routes"][0].get("distance") or 0) / 1000
    resolved_route = estimated_route or fallback
    return resolved_route, distance_km or estimated_distance_km


def get_route_provinces(origin: str, destination: str, waypoints: list[str] | None = None) -> tuple[list[str], float]:
    waypoints = combine_ordered_provinces(waypoints or [])
    stops = combine_ordered_provinces(origin, waypoints, destination)

    if len(stops) < 2:
        return combine_ordered_provinces(origin, waypoints, destination), 0

    observed_provinces: list[str] = []
    distance_km = 0.0

    for leg_origin, leg_destination in zip(stops, stops[1:]):
        leg_route, leg_distance_km = get_route_leg_provinces(leg_origin, leg_destination)
        observed_provinces = combine_ordered_provinces(observed_provinces, leg_route)
        distance_km += leg_distance_km

    route_result = build_sensible_route(origin, destination, waypoints, observed_provinces=observed_provinces)
    if len(route_result) < max(3, len(waypoints) + 2):
        route_result = combine_ordered_provinces(origin, observed_provinces, waypoints, destination)

    return route_result or combine_ordered_provinces(origin, waypoints, destination), distance_km


def retrieve_refs(question: str, top_k: int, embedding_model: str) -> list[dict[str, Any]]:
    rows = _state.get("rows") or []
    index = _state.get("index")
    if not rows or index is None:
        return []

    embedder = get_embedder(embedding_model)
    q_vec = embedder.encode([question], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    q_vec = normalize(q_vec)

    k = min(max(1, top_k), len(rows))
    scores, indices = index.search(q_vec, k)

    refs: list[dict[str, Any]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(rows):
            continue
        item = rows[idx]
        refs.append(
            {
                "score": float(score),
                "content": str(item.get("content") or ""),
                "row": int(item.get("row") or 0),
                "raw": item.get("std") or {},
            }
        )
    return refs


def fetch_docs_for_province(province: str, limit: int = 10) -> list[dict[str, Any]]:
    normalized = normalize_province_name(province)
    if not normalized:
        return []

    df = _state.get("df")
    if not isinstance(df, pd.DataFrame) or df.empty or "province" not in df.columns:
        return []

    province_df = df[df["province"].astype(str) == normalized].copy().head(limit)
    docs: list[dict[str, Any]] = []
    for _, row in province_df.iterrows():
        row_dict = {k: str(v) for k, v in row.to_dict().items()}
        docs.append({"content": row_to_text(row_dict), "raw": row_dict})
    return docs


def format_route_answer(
    origin: str,
    destination: str,
    provinces: list[str],
    dist_km: float,
    risk_points: list[dict[str, Any]],
    weather_info: str,
    removed_waypoint_notes: list[str] | None = None,
    origin_label: str | None = None,
    destination_label: str | None = None,
    waypoint_labels: list[str] | None = None,
) -> str:
    lines: list[str] = []
    distance_text = f"ประมาณ {dist_km:.0f} กม." if dist_km > 0 else "ไม่สามารถประเมินระยะทางได้"

    display_parts = [item for item in [origin_label or origin, *(waypoint_labels or []), destination_label or destination] if item]
    display_route = " -> ".join(display_parts) if display_parts else f"{origin} -> {destination}"

    lines.append(f"เส้นทาง: {display_route} ({distance_text})")
    lines.append(f"จังหวัดตามแนวเส้นทาง: {' -> '.join(provinces) if provinces else '-'}")

    if removed_waypoint_notes:
        lines.append("")
        lines.append("ระบบได้ปรับเส้นทางใหม่เนื่องจาก waypoint ที่ระบุไม่สมเหตุสมผล:")
        for note in removed_waypoint_notes:
            cleaned = note[2:] if isinstance(note, str) and note.startswith("- ") else note
            lines.append(f"- {cleaned}")

    lines.append("")
    lines.append("จุดเสี่ยงระหว่างทางที่ควรระวัง:")
    if risk_points:
        for idx, row in enumerate(risk_points, start=1):
            line = build_risk_point_line(row)
            line = line[2:] if line.startswith("- ") else line
            lines.append(f"{idx}. {line}")
    else:
        lines.append("- ไม่พบจุดเสี่ยงที่ระบุชัดเจนในข้อมูลเส้นทางนี้")

    if weather_info and weather_info != "ไม่มีข้อมูลสภาพอากาศ":
        lines.append("")
        lines.append("สภาพอากาศที่ควรเฝ้าระวัง:")
        for item in weather_info.splitlines():
            cleaned = item.strip()
            if cleaned:
                lines.append(cleaned)

    lines.append("")
    lines.append("คำแนะนำการขับขี่:")
    lines.append("- ชะลอความเร็วเมื่อเข้าใกล้ทางโค้ง ทางแยก และจุดกลับรถ")
    lines.append("- สังเกตป้ายเตือนและเตรียมเปลี่ยนเลนล่วงหน้าก่อนถึงจุดเสี่ยง")
    lines.append("- หากฝนตกหรือทัศนวิสัยไม่ดี ให้เพิ่มระยะห่างจากรถคันหน้า")
    return "\n".join(lines)


def is_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "none", "null", "nan", "ไม่ระบุ"}
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def first_present(*values: Any) -> Any:
    for value in values:
        if is_value_present(value):
            return value
    return None


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def deep_find_value(payload: Any, candidate_keys: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in candidate_keys and is_value_present(value):
                return value
        for value in payload.values():
            result = deep_find_value(value, candidate_keys)
            if is_value_present(result):
                return result
    elif isinstance(payload, list):
        for item in payload:
            result = deep_find_value(item, candidate_keys)
            if is_value_present(result):
                return result
    return None


def extract_event_coordinates(payload: Any) -> tuple[float | None, float | None]:
    lat = safe_float(deep_find_value(payload, {"latitude", "lat", "y"}))
    lng = safe_float(deep_find_value(payload, {"longitude", "lng", "lon", "x"}))
    if lat is not None and lng is not None:
        return lat, lng

    geometry = deep_find_value(payload, {"geometry", "shape"})
    if isinstance(geometry, dict):
        coordinates = geometry.get("coordinates")
        if isinstance(coordinates, list) and coordinates:
            first = coordinates[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                return safe_float(first[1]), safe_float(first[0])
            if isinstance(first, dict):
                lat = safe_float(first.get("lat"))
                lng = safe_float(first.get("lng") or first.get("lon"))
                if lat is not None and lng is not None:
                    return lat, lng

        links = geometry.get("links")
        if isinstance(links, list) and links:
            points = links[0].get("points") or []
            if points and isinstance(points[0], dict):
                lat = safe_float(points[0].get("lat"))
                lng = safe_float(points[0].get("lng"))
                if lat is not None and lng is not None:
                    return lat, lng

    return None, None


def parse_event_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat(sep=" ", timespec="seconds")
    except Exception:
        return text


def get_route_points(provinces: list[str], labels: list[str] | None = None) -> list[tuple[str, float, float]]:
    points: list[tuple[str, float, float]] = []

    for label in labels or []:
        lat, lng = geocode(label)
        if lat is not None and lng is not None:
            points.append((label, float(lat), float(lng)))

    for province in provinces:
        lat, lng = get_province_position(province)
        if lat is not None and lng is not None and province not in [item[0] for item in points]:
            points.append((province, float(lat), float(lng)))

    deduped: list[tuple[str, float, float]] = []
    seen_labels: set[str] = set()
    for label, lat, lng in points:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        deduped.append((label, lat, lng))
    return deduped


def build_route_bbox(route_points: list[tuple[str, float, float]], padding_deg: float = 0.12) -> str | None:
    if not route_points:
        return None
    lats = [lat for _, lat, _ in route_points]
    lngs = [lng for _, _, lng in route_points]
    min_lat = min(lats) - padding_deg
    max_lat = max(lats) + padding_deg
    min_lng = min(lngs) - padding_deg
    max_lng = max(lngs) + padding_deg
    return f"{min_lng},{min_lat},{max_lng},{max_lat}"


def fetch_realtime_events(max_age_seconds: int = 600) -> list[dict[str, Any]]:
    now = time.time()
    cached_age = now - float(_realtime_cache.get("fetched_at") or 0)
    if _realtime_cache.get("events") and cached_age <= max_age_seconds:
        return list(_realtime_cache["events"])

    try:
        response = http.get(LONGDO_EVENT_URL, timeout=8)
        response.raise_for_status()
        data = response.json()
        events = data if isinstance(data, list) else []
        normalized_events: list[dict[str, Any]] = []
        for item in events:
            try:
                normalized_events.append(
                    {
                        "source": "Longdo/iTIC",
                        "source_id": str(item.get("eid") or item.get("id") or ""),
                        "title": str(item.get("title") or item.get("title_en") or "เหตุจราจร"),
                        "description": str(item.get("description") or item.get("description_en") or "").strip(),
                        "lat": float(item.get("latitude")) if item.get("latitude") not in (None, "") else None,
                        "lng": float(item.get("longitude")) if item.get("longitude") not in (None, "") else None,
                        "start": parse_event_time(item.get("start")),
                        "severity": str(item.get("severity") or ""),
                        "icon": str(item.get("icon") or ""),
                    }
                )
            except Exception:
                continue
        _realtime_cache["fetched_at"] = now
        _realtime_cache["events"] = normalized_events
        return normalized_events
    except Exception:
        return list(_realtime_cache.get("events") or [])


def fetch_exat_events_for_route(route_points: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    if not EXAT_API_BASE_URL:
        return []

    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    year = datetime.now().year
    month = datetime.now().month

    for _ in range(EXAT_LOOKBACK_MONTHS):
        for endpoint in ["EXAT_Accident", "EXAT_Crash"]:
            url = f"{EXAT_API_BASE_URL.rstrip('/')}/{endpoint}/{year}/{month}"
            try:
                response = http.get(url, timeout=8)
                response.raise_for_status()
                payload = response.json()
            except Exception:
                continue

            for item in payload.get("result") or []:
                lat, lng = extract_event_coordinates(item)
                title = first_present(
                    deep_find_value(item, {"title", "event_name", "accident_type", "crash_type", "type"}),
                    endpoint.replace("EXAT_", "EXAT "),
                )
                road_name = first_present(deep_find_value(item, {"road", "road_name", "route", "expressway", "location"}), "")
                description = first_present(
                    deep_find_value(item, {"description", "detail", "remark", "cause", "location"}),
                    road_name,
                    title,
                )
                start = parse_event_time(first_present(deep_find_value(item, {"date", "datetime", "start_time", "created_at"}), f"{year}-{month:02d}-01"))
                source_id = str(first_present(deep_find_value(item, {"id", "accident_id", "crash_id"}), f"{endpoint}-{year}-{month}-{len(events)}"))
                if source_id in seen:
                    continue
                seen.add(source_id)
                events.append(
                    {
                        "source": "EXAT",
                        "source_id": source_id,
                        "title": str(title or "เหตุบนทางพิเศษ"),
                        "description": str(description or road_name or "ข้อมูลอุบัติเหตุจาก EXAT"),
                        "lat": lat,
                        "lng": lng,
                        "start": start,
                        "severity": "",
                        "icon": endpoint.lower(),
                    }
                )

        month -= 1
        if month <= 0:
            month = 12
            year -= 1

    return events[:20]


def fetch_here_events_for_route(route_points: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    if not HERE_API_KEY or not route_points:
        return []

    query_points = route_points if len(route_points) <= 3 else [route_points[0], route_points[len(route_points) // 2], route_points[-1]]
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    radius_m = max(8000, int(REALTIME_ROUTE_RADIUS_KM * 1000))

    for _, lat, lng in query_points:
        try:
            response = http.get(
                HERE_TRAFFIC_URL,
                params={
                    "in": f"circle:{lat},{lng};r={radius_m}",
                    "locationReferencing": "none",
                    "lang": "th-TH",
                    "apiKey": HERE_API_KEY,
                },
                timeout=8,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue

        for item in payload.get("results") or []:
            details = item.get("incidentDetails") or item
            source_id = str(first_present(details.get("id"), item.get("id"), f"here-{len(events)}"))
            if source_id in seen:
                continue
            seen.add(source_id)

            lat_value, lng_value = extract_event_coordinates(item)
            description = first_present(
                (details.get("description") or {}).get("value") if isinstance(details.get("description"), dict) else details.get("description"),
                (details.get("summary") or {}).get("value") if isinstance(details.get("summary"), dict) else details.get("summary"),
                deep_find_value(details.get("events") or details, {"description"}),
            )
            title = first_present(details.get("type"), details.get("criticality"), "HERE incident")
            road_from = first_present(details.get("from"), deep_find_value(details, {"roadname", "roadnumbers"}), "")
            if road_from and description:
                description = f"{description} | {road_from}"

            events.append(
                {
                    "source": "HERE",
                    "source_id": source_id,
                    "title": str(title or "HERE incident"),
                    "description": str(description or "ข้อมูลเหตุจราจรจาก HERE"),
                    "lat": lat_value,
                    "lng": lng_value,
                    "start": parse_event_time(first_present(details.get("startTime"), details.get("lastReportTime"))),
                    "severity": str(first_present(details.get("criticality"), details.get("severity"), "")),
                    "icon": str(first_present(details.get("type"), "here")),
                }
            )

    return events[:20]


def fetch_tomtom_events_for_route(route_points: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    if not TOMTOM_API_KEY or not route_points:
        return []

    bbox = build_route_bbox(route_points)
    if not bbox:
        return []

    try:
        response = http.get(
            TOMTOM_TRAFFIC_URL,
            params={
                "key": TOMTOM_API_KEY,
                "bbox": bbox,
                "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code,iconCategory},startTime,endTime,from,to,roadNumbers,length,delay,timeValidity}}}",
                "language": "th-TH",
                "timeValidityFilter": "present",
            },
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    events: list[dict[str, Any]] = []
    for item in payload.get("incidents") or []:
        props = item.get("properties") or {}
        event_list = props.get("events") or []
        first_event = event_list[0] if event_list else {}
        lat, lng = extract_event_coordinates(item)
        description = first_present(first_event.get("description"), props.get("from"), props.get("to"), "ข้อมูลเหตุจราจรจาก TomTom")
        road_numbers = props.get("roadNumbers") or []
        road_text = ", ".join(road_numbers) if isinstance(road_numbers, list) else str(road_numbers or "")
        if road_text and description:
            description = f"{description} | ถนน {road_text}"
        title = first_present(first_event.get("description"), props.get("iconCategory"), "TomTom incident")

        events.append(
            {
                "source": "TomTom",
                "source_id": str(first_present(props.get("id"), f"tomtom-{len(events)}")),
                "title": str(title or "TomTom incident"),
                "description": str(description or "ข้อมูลเหตุจราจรจาก TomTom"),
                "lat": lat,
                "lng": lng,
                "start": parse_event_time(props.get("startTime")),
                "severity": str(first_present(props.get("magnitudeOfDelay"), props.get("iconCategory"), "")),
                "icon": str(first_present(props.get("iconCategory"), item.get("type"), "tomtom")),
            }
        )

    return events[:20]


def merge_multisource_realtime_events(provinces: list[str], labels: list[str] | None = None) -> list[dict[str, Any]]:
    route_points = get_route_points(provinces, labels)
    events: list[dict[str, Any]] = []
    events.extend(fetch_realtime_events())
    events.extend(fetch_exat_events_for_route(route_points))
    events.extend(fetch_here_events_for_route(route_points))
    events.extend(fetch_tomtom_events_for_route(route_points))

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if not is_traffic_related_event(event):
            continue
        key = "|".join(
            [
                str(event.get("source") or ""),
                str(event.get("source_id") or ""),
                str(event.get("title") or "")[:80].lower(),
                str(event.get("start") or ""),
                str(round(float(event.get("lat") or 0), 3)),
                str(round(float(event.get("lng") or 0), 3)),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def is_traffic_related_event(event: dict[str, Any]) -> bool:
    text = f"{event.get('title', '')} {event.get('description', '')} {event.get('icon', '')}".lower()
    traffic_keywords = [
        "อุบัติเหตุ",
        "รถเสีย",
        "ชน",
        "กีดขวาง",
        "จราจร",
        "ติดขัด",
        "ปิดเบี่ยง",
        "accident",
        "breakdown",
        "crash",
        "traffic",
        "jam",
        "congestion",
        "road",
        "lane",
    ]
    blocked_keywords = ["เพลิงไหม้", "fire", "flood rescue", "earthquake"]
    return any(keyword in text for keyword in traffic_keywords) and not any(keyword in text for keyword in blocked_keywords)


def event_priority_score(event: dict[str, Any]) -> int:
    text = f"{event.get('title', '')} {event.get('description', '')} {event.get('icon', '')}".lower()
    score = 0

    source = str(event.get("source") or "")
    if source == "EXAT":
        score += 5
    elif source in {"HERE", "TomTom"}:
        score += 4
    elif source == "Longdo/iTIC":
        score += 2

    if any(keyword in text for keyword in ["อุบัติเหตุ", "รถเสีย", "ชน", "กีดขวาง", "accident", "breakdown", "crash"]):
        score += 10
    if any(keyword in text for keyword in ["traffic", "jam", "congestion", "ปิดการจราจร", "จราจร", "ติดขัด"]):
        score += 4

    severity = safe_float(event.get("severity"))
    if severity is not None:
        score += int(severity)

    start_text = str(event.get("start") or "").strip()
    if start_text:
        try:
            parsed = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
            now = datetime.now(parsed.tzinfo) if parsed.tzinfo else datetime.now()
            age_hours = abs((now - parsed).total_seconds()) / 3600
            if age_hours <= 6:
                score += 4
            elif age_hours <= 24:
                score += 2
            elif age_hours > 72 and "roadworks" not in text:
                score -= 2
        except Exception:
            pass

    return score


def point_to_segment_distance_km(
    lat: float,
    lng: float,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
) -> float:
    mean_lat = radians((start_lat + end_lat + lat) / 3)
    km_per_deg_lat = 111.32
    km_per_deg_lng = 111.32 * cos(mean_lat)

    px = lng * km_per_deg_lng
    py = lat * km_per_deg_lat
    ax = start_lng * km_per_deg_lng
    ay = start_lat * km_per_deg_lat
    bx = end_lng * km_per_deg_lng
    by = end_lat * km_per_deg_lat

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return sqrt((px - ax) ** 2 + (py - ay) ** 2)

    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def filter_events_near_route(
    provinces: list[str], labels: list[str] | None = None, radius_km: float = REALTIME_ROUTE_RADIUS_KM
) -> list[dict[str, Any]]:
    events = merge_multisource_realtime_events(provinces, labels)
    if not events:
        return []

    route_points = [(lat, lng) for _, lat, lng in get_route_points(provinces, labels)]
    if len(route_points) < 2:
        ranked = sorted(events, key=event_priority_score, reverse=True)
        return ranked[:5]

    matched: list[dict[str, Any]] = []
    for event in events:
        lat = event.get("lat")
        lng = event.get("lng")
        if lat is None or lng is None:
            continue

        best_distance = float("inf")
        for (start_lat, start_lng), (end_lat, end_lng) in zip(route_points, route_points[1:]):
            distance = point_to_segment_distance_km(float(lat), float(lng), start_lat, start_lng, end_lat, end_lng)
            best_distance = min(best_distance, distance)

        if best_distance <= radius_km:
            matched.append({**event, "route_distance_km": round(best_distance, 1)})

    matched = sorted(
        matched,
        key=lambda item: (-event_priority_score(item), float(item.get("route_distance_km") or 999), str(item.get("start") or "")),
    )
    relevant = [event for event in matched if event_priority_score(event) > 0]
    return (relevant or matched)[:8]


def weather_code_to_text(code: int | None) -> str:
    mapping = {
        0: "ท้องฟ้าโปร่ง",
        1: "แดดจัด",
        2: "มีเมฆบางส่วน",
        3: "เมฆมาก",
        45: "มีหมอก",
        48: "หมอกจัด",
        51: "ฝนละอองเบา",
        53: "ฝนละอองปานกลาง",
        55: "ฝนละอองหนัก",
        61: "ฝนเบา",
        63: "ฝนปานกลาง",
        65: "ฝนหนัก",
        80: "ฝนซู่เบา",
        81: "ฝนซู่ปานกลาง",
        82: "ฝนซู่หนัก",
        95: "พายุฝนฟ้าคะนอง",
    }
    return mapping.get(int(code or 0), "สภาพอากาศทั่วไป")


def fetch_current_weather_for_route(
    provinces: list[str], labels: list[str] | None = None, max_points: int = 3, max_age_seconds: int = 900
) -> list[dict[str, Any]]:
    deduped = get_route_points(provinces, labels)
    if not deduped:
        return []

    if len(deduped) > max_points:
        middle = deduped[len(deduped) // 2]
        deduped = [deduped[0], middle, deduped[-1]] if len(deduped) >= 3 else deduped[:max_points]

    cache_key = "|".join([f"{label}:{lat:.3f}:{lng:.3f}" for label, lat, lng in deduped])
    now = time.time()
    cache_items = _weather_cache.get("items") or {}
    cached = cache_items.get(cache_key)
    if cached and now - float(cached.get("fetched_at") or 0) <= max_age_seconds:
        return list(cached.get("snapshots") or [])

    try:
        latitudes = ",".join([str(lat) for _, lat, _ in deduped])
        longitudes = ",".join([str(lng) for _, _, lng in deduped])
        response = http.get(
            OPEN_METEO_URL,
            params={
                "latitude": latitudes,
                "longitude": longitudes,
                "current": "temperature_2m,precipitation,rain,weather_code,wind_speed_10m",
            },
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload if isinstance(payload, list) else [payload]

        snapshots: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            current = item.get("current") or {}
            label = deduped[idx][0] if idx < len(deduped) else f"จุดที่ {idx + 1}"
            snapshots.append(
                {
                    "label": label,
                    "time": str(current.get("time") or ""),
                    "temperature_c": current.get("temperature_2m"),
                    "rain_mm": current.get("rain") if current.get("rain") is not None else current.get("precipitation"),
                    "precipitation_mm": current.get("precipitation"),
                    "wind_kmh": current.get("wind_speed_10m"),
                    "weather_code": current.get("weather_code"),
                    "condition": weather_code_to_text(current.get("weather_code")),
                }
            )

        cache_items[cache_key] = {"fetched_at": now, "snapshots": snapshots}
        _weather_cache["items"] = cache_items
        _weather_cache["fetched_at"] = now
        return snapshots
    except Exception:
        return list(cached.get("snapshots") or []) if cached else []


def build_realtime_fallback(
    route_title: str,
    provinces: list[str],
    dist_km: float,
    nearby_events: list[dict[str, Any]],
    current_weather: list[dict[str, Any]],
    history_summary: list[str],
    risk_points: list[dict[str, Any]],
    weather_info: str,
    web_summary: str = "",
    sources_checked: list[str] | None = None,
) -> str:
    lines: list[str] = [f"อัปเดตเหตุจราจรล่าสุดสำหรับเส้นทาง: {route_title}"]
    if provinces:
        lines.append(f"แนวเส้นทางหลัก: {' -> '.join(provinces)}")
    if dist_km > 0:
        lines.append(f"ระยะทางโดยประมาณ: {dist_km:.0f} กม.")

    live_sources = sources_checked or sorted({str(event.get('source') or '') for event in nearby_events if event.get('source')})
    if live_sources:
        lines.append(f"แหล่งข้อมูลสดที่ตรวจสอบ: {', '.join(live_sources)}")

    lines.append("")
    if nearby_events:
        lines.append(f"เหตุการณ์สดใกล้แนวเส้นทางพบ {len(nearby_events)} จุด:")
        for idx, event in enumerate(nearby_events[:4], start=1):
            detail = str(event.get("description") or event.get("title") or "").replace("\r", " ").replace("\n", " ").strip()
            if len(detail) > 180:
                detail = detail[:177] + "..."
            when = f" | เวลา {event['start']}" if event.get("start") else ""
            distance_note = f" | ใกล้แนวเส้นทางประมาณ {event['route_distance_km']} กม." if event.get("route_distance_km") is not None else ""
            source = f"[{event.get('source', 'source')}] " if event.get('source') else ""
            lines.append(f"{idx}. {source}{event.get('title', 'เหตุจราจร')} — {detail}{distance_note}{when}")
    else:
        lines.append("เหตุการณ์สด: ยังไม่พบอุบัติเหตุหรือเหตุจราจรเด่นใกล้แนวเส้นทางจาก feed ล่าสุด")

    if web_summary:
        lines.append("")
        lines.append("ข้อมูลเว็บสาธารณะเพิ่มเติมผ่าน OpenAI:")
        for item in str(web_summary).splitlines()[:6]:
            if item.strip():
                lines.append(item.strip())

    if current_weather:
        lines.append("")
        lines.append("สภาพอากาศล่าสุดตามแนวเส้นทาง:")
        for item in current_weather[:3]:
            rain = float(item.get("rain_mm") or 0)
            rain_note = f", ฝน {rain:.1f} มม." if rain > 0 else ""
            wind = item.get("wind_kmh")
            wind_note = f", ลม {float(wind):.0f} กม./ชม." if wind is not None else ""
            temp = item.get("temperature_c")
            temp_note = f"{float(temp):.1f}°C" if temp is not None else "ไม่ทราบอุณหภูมิ"
            lines.append(f"- {item['label']}: {item['condition']} {temp_note}{rain_note}{wind_note}")

    if risk_points:
        lines.append("")
        lines.append("จุดเสี่ยงจากสถิติย้อนหลัง:")
        for idx, row in enumerate(risk_points[:4], start=1):
            line = build_risk_point_line(row)
            lines.append(f"{idx}. {line[2:] if line.startswith('- ') else line}")

    if history_summary:
        lines.append("")
        lines.append("ภาพรวมจาก dataset ย้อนหลัง:")
        for summary in history_summary[:3]:
            lines.append(summary)

    if weather_info and weather_info != "ไม่มีข้อมูลสภาพอากาศ":
        lines.append("")
        lines.append("สภาพอากาศที่เคยพบร่วมกับอุบัติเหตุ:")
        for item in weather_info.splitlines()[:3]:
            if item.strip():
                lines.append(item.strip())

    lines.append("")
    source_summary = ", ".join(live_sources + ["Open-Meteo", "RoadBot dataset"]) if live_sources else "Longdo/iTIC, Open-Meteo, RoadBot dataset"
    lines.append(f"แหล่งข้อมูล: {source_summary}")
    return "\n".join(lines)


def answer_realtime_with_llm(question: str, context: str, fallback: str) -> str:
    prompt = (
        "คุณคือ RoadBot AI ผู้ช่วยข้อมูลอุบัติเหตุและการเดินทางในไทย\n"
        "จงสรุปคำตอบแบบมืออาชีพ เป็นภาษาไทย ใช้เฉพาะข้อมูลที่ให้มาเท่านั้น\n"
        "ต้องตอบให้ครบและอ่านง่าย โดยเรียงหัวข้อดังนี้:\n"
        "1) สรุปตอนนี้\n2) เหตุการณ์สดตามเส้นทาง\n3) จุดเสี่ยงจากสถิติย้อนหลัง\n4) คำแนะนำการเดินทาง\n5) แหล่งข้อมูล\n"
        "ห้ามแต่งข้อมูลเพิ่ม และถ้าไม่พบเหตุการณ์สดให้บอกตรง ๆ\n\n"
        f"Context:\n{context}\n\n"
        f"คำถาม: {question}\nคำตอบ:"
    )
    answer = llm_invoke(prompt)
    return answer if answer else fallback


def format_realtime_route_answer(question: str, route_context: dict[str, Any]) -> str:
    origin = route_context.get("origin")
    destination = route_context.get("destination")
    waypoints = route_context.get("waypoints") or []
    labels = route_context.get("place_labels") or []

    if origin and destination:
        provinces, dist_km = get_route_provinces(origin, destination, waypoints)
        nearby_events = filter_events_near_route(provinces, labels)
        risk_points = extract_route_risk_points(provinces, limit_per_province=2, total_limit=6)
        weather_info = get_weather_summary(provinces)
        current_weather = fetch_current_weather_for_route(provinces, labels)
        history_summary = [build_province_stats_summary(province) for province in provinces[:3]]
        route_title = " -> ".join(labels or [origin, destination])
        web_summary = fetch_openai_web_realtime_summary(question, route_title, provinces, nearby_events, current_weather)
        sources_checked = ["Longdo/iTIC", "EXAT"]
        if web_summary:
            sources_checked.append("OpenAI Web Search")
        if HERE_API_KEY:
            sources_checked.append("HERE")
        if TOMTOM_API_KEY:
            sources_checked.append("TomTom")

        event_lines = []
        for idx, event in enumerate(nearby_events[:5], start=1):
            detail = str(event.get("description") or event.get("title") or "").replace("\r", " ").replace("\n", " ").strip()
            event_lines.append(
                f"- {idx}. [{event.get('source', 'source')}] {event.get('title', 'เหตุจราจร')} | รายละเอียด: {detail} | เวลา: {event.get('start') or '-'} | ระยะห่างจากแนวเส้นทางประมาณ: {event.get('route_distance_km', '-')} กม."
            )

        weather_lines = []
        for item in current_weather:
            weather_lines.append(
                f"- {item['label']}: {item['condition']}, อุณหภูมิ {item.get('temperature_c', '-') }°C, ฝน {item.get('rain_mm', 0)} มม., ลม {item.get('wind_kmh', '-') } กม./ชม."
            )

        risk_lines = [build_risk_point_line(row) for row in risk_points[:5]]
        history_lines = history_summary[:3]

        context = (
            f"เส้นทางที่ถาม: {route_title}\n"
            f"จังหวัดตามแนวเส้นทาง: {' -> '.join(provinces) if provinces else '-'}\n"
            f"ระยะทางโดยประมาณ: {dist_km:.0f} กม.\n"
            f"แหล่งข้อมูลสดที่ตรวจสอบ: {', '.join(sources_checked)}\n\n"
            f"เหตุการณ์สดจากหลายแหล่ง:\n{chr(10).join(event_lines) if event_lines else '- ไม่พบเหตุการณ์เด่นใกล้แนวเส้นทาง'}\n\n"
            f"สรุปจาก OpenAI Web Search:\n{web_summary or '- ไม่พบข้อมูลเว็บเพิ่มเติมที่เชื่อถือได้'}\n\n"
            f"สภาพอากาศล่าสุดจาก Open-Meteo:\n{chr(10).join(weather_lines) if weather_lines else '- ไม่มีข้อมูลสภาพอากาศล่าสุด'}\n\n"
            f"จุดเสี่ยงจาก dataset ย้อนหลัง:\n{chr(10).join(risk_lines) if risk_lines else '- ไม่พบจุดเสี่ยงเด่น'}\n\n"
            f"สรุปสถิติย้อนหลังรายจังหวัด:\n{chr(10).join(history_lines) if history_lines else '- ไม่มีข้อมูลสรุปเพิ่มเติม'}\n\n"
            f"สภาพอากาศในอดีตที่พบบ่อยร่วมกับอุบัติเหตุ:\n{weather_info or 'ไม่มีข้อมูล'}"
        )

        fallback = build_realtime_fallback(
            route_title=route_title,
            provinces=provinces,
            dist_km=dist_km,
            nearby_events=nearby_events,
            current_weather=current_weather,
            history_summary=history_summary,
            risk_points=risk_points,
            weather_info=weather_info,
            web_summary=web_summary,
            sources_checked=sources_checked,
        )
        return answer_realtime_with_llm(question, context, fallback)

    events = fetch_realtime_events()
    if not events:
        return "ยังดึงข้อมูล real-time ไม่สำเร็จในขณะนี้ แต่ dataset ย้อนหลังยังใช้งานได้ตามปกติ"

    lines = ["อัปเดตเหตุจราจรล่าสุดจาก Longdo/iTIC feed:"]
    for idx, event in enumerate(sorted(events, key=event_priority_score, reverse=True)[:5], start=1):
        detail = event.get("description") or event.get("title")
        when = f" | เวลา {event['start']}" if event.get("start") else ""
        lines.append(f"{idx}. {event.get('title', 'เหตุจราจร')} — {detail}{when}")
    lines.append("")
    lines.append("แหล่งข้อมูล: Longdo/iTIC realtime feed")
    return "\n".join(lines)


def deterministic_fallback(question: str, refs: list[dict[str, Any]]) -> str:
    if not refs:
        return "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้จากฐานข้อมูล"

    lines = [f"สรุปจากข้อมูลที่ค้นได้ {len(refs)} รายการ"]
    for item in refs[:6]:
        raw = item.get("raw") or {}
        province = raw.get("province", "ไม่ระบุจังหวัด")
        road = raw.get("road_name", "ไม่ระบุสายทาง")
        cause = raw.get("cause", "ไม่ระบุ")
        lines.append(f"- {province} | {road} | สาเหตุ: {cause}")

    lines.append("")
    lines.append("คำแนะนำ: ขับด้วยความเร็วเหมาะสม เว้นระยะห่าง และระวังบริเวณทางแยก/ทางโค้ง")
    return "\n".join(lines)


def answer_with_llm(question: str, context: str, fallback: str) -> str:
    prompt = (
        "คุณคือ Roadbot ผู้ช่วยข้อมูลอุบัติเหตุในประเทศไทย\n"
        "ตอบจากข้อมูลที่มีเท่านั้น ตอบเป็นภาษาไทย ตอบครั้งเดียวไม่ทวนซ้ำ\n"
        "ห้ามแต่งข้อมูลใหม่ที่ไม่มีใน context\n\n"
        f"Context:\n{context if context else 'ไม่มีข้อมูลเพียงพอ'}\n\n"
        f"คำถาม: {question}\n"
        "คำตอบ:"
    )
    answer = llm_invoke(prompt)
    return answer if answer else fallback


def ask_roadbot(question: str, top_k: int, embedding_model: str) -> tuple[str, list[dict[str, Any]]]:
    q_type = classify_question(question)

    if q_type == "irrelevant":
        return (
            "ขออภัยค่ะ ระบบนี้เน้นตอบเรื่องอุบัติเหตุ เส้นทาง และข้อมูลจราจรที่เกี่ยวข้องเท่านั้น",
            [],
        )

    route_context = build_route_context(question)

    if q_type == "realtime":
        refs: list[dict[str, Any]] = []
        for province in route_context.get("waypoints") or []:
            refs.extend(fetch_docs_for_province(province, limit=4))
        if route_context.get("origin"):
            refs.extend(fetch_docs_for_province(route_context["origin"], limit=4))
        if route_context.get("destination"):
            refs.extend(fetch_docs_for_province(route_context["destination"], limit=6))
        return format_realtime_route_answer(question, route_context), refs[:20]

    if q_type == "route_analysis":
        origin = route_context.get("origin")
        destination = route_context.get("destination")
        waypoints = route_context.get("waypoints") or []
        if origin and destination:
            filtered_waypoints, removed = filter_unreasonable_waypoints(
                origin,
                destination,
                waypoints,
                return_removed=True,
            )
            provinces, dist_km = get_route_provinces(origin, destination, filtered_waypoints)
            if not provinces:
                provinces = combine_ordered_provinces(origin, filtered_waypoints, destination)

            risk_points = extract_route_risk_points(provinces, limit_per_province=2, total_limit=10)
            weather_info = get_weather_summary(provinces)
            answer = format_route_answer(
                origin=origin,
                destination=destination,
                provinces=provinces,
                dist_km=dist_km,
                risk_points=risk_points,
                weather_info=weather_info,
                removed_waypoint_notes=removed,
                origin_label=route_context.get("origin_label"),
                destination_label=route_context.get("destination_label"),
                waypoint_labels=route_context.get("waypoint_labels") or [],
            )

            docs: list[dict[str, Any]] = []
            for province in provinces:
                docs.extend(fetch_docs_for_province(province, limit=6))
            return answer, docs[:40]

    refs: list[dict[str, Any]] = []

    if q_type in {"top_provinces", "top_causes", "other_accident_stats"}:
        refs = retrieve_refs(question, max(top_k, 10), embedding_model)

    if q_type == "top_provinces":
        df = _state.get("df")
        if isinstance(df, pd.DataFrame) and not df.empty and "province" in df.columns:
            top_provinces_df = df["province"].value_counts().reset_index().head(5)
            top_provinces_df.columns = ["province", "accident_count"]
            top_provinces_df = top_provinces_df[top_provinces_df["province"] != "ไม่ระบุ"]
            top_str = ", ".join(
                [
                    f"{row['province']} ({int(row['accident_count']):,} ครั้ง)"
                    for _, row in top_provinces_df.iterrows()
                ]
            )
            fallback = f"จังหวัดที่พบอุบัติเหตุสูงสุด: {top_str}" if top_str else "ไม่มีข้อมูลเพียงพอ"
            return answer_with_llm(question, top_str, fallback), refs

    if q_type == "top_causes":
        df = _state.get("df")
        if isinstance(df, pd.DataFrame) and not df.empty and "cause" in df.columns:
            top_causes_df = df["cause"].value_counts().reset_index().head(5)
            top_causes_df.columns = ["cause", "accident_count"]
            top_causes_df = top_causes_df[top_causes_df["cause"] != "ไม่ระบุ"]
            top_str = ", ".join(
                [f"{row['cause']} ({int(row['accident_count']):,} ครั้ง)" for _, row in top_causes_df.iterrows()]
            )
            fallback = f"สาเหตุอุบัติเหตุที่พบบ่อย: {top_str}" if top_str else "ไม่มีข้อมูลเพียงพอ"
            return answer_with_llm(question, top_str, fallback), refs

    if q_type == "province_details":
        provinces = extract_provinces_from_text(question)
        province_name = provinces[0] if provinces else None
        if province_name:
            province_docs = fetch_docs_for_province(province_name, limit=12)
            province_stats = build_province_stats_summary(province_name)
            weather = get_weather_summary([province_name])
            context = (
                f"สถิติจังหวัด:\n{province_stats}\n\n"
                f"สภาพอากาศที่พบ:\n{weather}\n\n"
                f"ตัวอย่างเหตุการณ์:\n" + "\n".join([f"- {item['content']}" for item in province_docs[:8]])
            )
            fallback = province_stats
            return answer_with_llm(question, context, fallback), province_docs

    overall = build_overall_stats_summary()
    context = overall + "\n\n" + "\n".join([f"- {item.get('content', '')}" for item in refs[:10]])
    fallback = deterministic_fallback(question, refs)
    return answer_with_llm(question, context, fallback), refs


def warmup_default_resources() -> dict[str, Any]:
    embedding_model = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    index_path = resolve_data_path(os.getenv("FAISS_INDEX_PATH", "./data/faiss.index"))
    meta_path = resolve_data_path(os.getenv("FAISS_META_PATH", "./data/faiss_meta.json"))

    _warmup_state["last_error"] = ""

    try:
        get_embedder(embedding_model)
        _warmup_state["embedder_ready"] = True
    except Exception as exc:
        _warmup_state["embedder_ready"] = False
        _warmup_state["last_error"] = f"embedder load failed: {exc}"

    ready = ensure_runtime_ready(embedding_model, index_path, meta_path)

    _warmup_state["index_ready"] = bool(_state.get("index") is not None or os.path.exists(index_path))
    _warmup_state["meta_ready"] = bool((_state.get("rows") or []) or os.path.exists(meta_path))
    _warmup_state["ready"] = bool(ready and _warmup_state["embedder_ready"])
    _warmup_state["updated_at"] = int(time.time())

    if not _warmup_state["ready"] and not _warmup_state["last_error"]:
        _warmup_state["last_error"] = "warmup incomplete"

    get_llm()

    return {
        "ok": True,
        "ready": _warmup_state["ready"],
        "embedder_ready": _warmup_state["embedder_ready"],
        "index_ready": _warmup_state["index_ready"],
        "meta_ready": _warmup_state["meta_ready"],
        "last_error": _warmup_state["last_error"],
        "index_path": index_path,
        "meta_path": meta_path,
    }


@app.on_event("startup")
def on_startup() -> None:
    warmup_default_resources()


@app.get("/status")
def status() -> dict[str, Any]:
    return {
        "ok": True,
        "ready": _warmup_state["ready"],
        "embedder_ready": _warmup_state["embedder_ready"],
        "index_ready": _warmup_state["index_ready"],
        "meta_ready": _warmup_state["meta_ready"],
        "last_error": _warmup_state["last_error"],
        "updated_at": _warmup_state["updated_at"],
        "index_path": resolve_data_path(os.getenv("FAISS_INDEX_PATH", "./data/faiss.index")),
        "meta_path": resolve_data_path(os.getenv("FAISS_META_PATH", "./data/faiss_meta.json")),
        "groq_ready": bool(get_llm() is not None),
        "llm_provider": _state.get("llm_provider", "none"),
    }


@app.post("/warmup")
def warmup() -> dict[str, Any]:
    return warmup_default_resources()


@app.post("/ingest-sheet")
def ingest_sheet(body: IngestBody) -> dict[str, Any]:
    try:
        index_path = resolve_data_path(body.index_path)
        meta_path = resolve_data_path(body.meta_path)
        payload = prepare_runtime(
            sheet_url=body.sheet_url,
            gid=body.gid,
            embedding_model=body.embedding_model,
            index_path=index_path,
            meta_path=meta_path,
        )

        _warmup_state["ready"] = True
        _warmup_state["embedder_ready"] = True
        _warmup_state["index_ready"] = True
        _warmup_state["meta_ready"] = True
        _warmup_state["last_error"] = ""
        _warmup_state["updated_at"] = int(time.time())

        return {
            "ok": True,
            "count": len(payload.get("rows") or []),
            "index_path": index_path,
            "meta_path": meta_path,
            "index_built": True,
            "source": payload.get("source") or {},
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/chat")
def chat(body: ChatBody) -> dict[str, Any]:
    try:
        index_path = resolve_data_path(body.index_path)
        meta_path = resolve_data_path(body.meta_path)

        if not ensure_runtime_ready(body.embedding_model, index_path, meta_path):
            raise ValueError("Index/meta files not found. Please ingest sheet first.")

        answer, refs = ask_roadbot(body.question, top_k=max(1, body.top_k), embedding_model=body.embedding_model)
        return {"ok": True, "answer": answer, "references": refs}
    except Exception as exc:
        return {
            "ok": True,
            "answer": f"ระบบยังไม่พร้อมใช้งานชั่วคราว ({str(exc)}). กรุณาลองใหม่อีกครั้งใน 1-2 นาที",
            "references": [],
        }

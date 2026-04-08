import json
import os
import re
import time
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

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

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

_state: dict[str, Any] = {
    "rows": [],
    "texts": [],
    "df": pd.DataFrame(),
    "weather_summary": {},
    "llm": None,
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


def get_llm() -> ChatGroq | None:
    if _state.get("llm") is not None:
        return _state["llm"]

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        _state["llm"] = None
        return None

    model = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0.1"))

    try:
        _state["llm"] = ChatGroq(model=model, temperature=temperature, api_key=api_key)
    except Exception:
        _state["llm"] = None
    return _state["llm"]


def llm_invoke(prompt: str) -> str:
    llm = get_llm()
    if llm is None:
        return ""

    try:
        return str(llm.invoke(prompt).content or "").strip()
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


def classify_question(question: str) -> str:
    question = str(question or "").strip()
    found_provinces = extract_provinces_from_text(question)

    if any(hint in question for hint in IRRELEVANT_HINTS):
        return "irrelevant"

    if any(word in question for word in ["ตอนนี้", "ล่าสุด", "เรียลไทม์", "สด"]):
        return "realtime"

    if any(
        word in question
        for word in ["จังหวัดที่มีอุบัติเหตุ", "จังหวัดไหนมีอุบัติเหตุ", "มากสุด", "สูงสุด", "มากที่สุด"]
    ):
        return "top_provinces"

    if any(word in question for word in ["สาเหตุ", "ต้นเหตุ", "เกิดจากอะไร"]):
        return "top_causes"

    if len(found_provinces) >= 2 and any(
        word in question for word in ["ไป", "จาก", "ผ่าน", "เส้นทาง", "จุดเสี่ยง", "ระวัง", "ขับ"]
    ):
        return "route_analysis"

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
    normalized = normalize_province_name(place_name) or place_name
    queries = [
        f"{normalized}, Thailand",
        f"จังหวัด{normalized}, Thailand",
        str(place_name),
    ]

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
    orig_lat, orig_lng = geocode(origin)
    wp_lat, wp_lng = geocode(waypoint)
    dest_lat, dest_lng = geocode(destination)

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

    base_route = infer_route_provinces_with_llm(origin, destination, [], include_waypoints=False)
    filtered: list[str] = []
    removed_messages: list[str] = []
    current_origin = origin

    for waypoint in waypoints:
        keep = True
        reason = ""
        direct_km, via_km, extra_km = detour_metrics(current_origin, waypoint, destination)
        route_with_waypoint = infer_route_provinces_with_llm(
            current_origin,
            destination,
            [waypoint],
            include_waypoints=True,
            force_include_waypoints=False,
        )

        if direct_km is not None and via_km is not None and extra_km is not None:
            ratio = via_km / direct_km if direct_km > 0 else 999
            if via_km > direct_km * 1.45 and extra_km > 180:
                keep = False
                reason = f"ทำให้เส้นทางอ้อมเพิ่มประมาณ {extra_km:.0f} กม. ({ratio:.2f} เท่าของเส้นทางตรง)"
            elif base_route and waypoint not in base_route and route_with_waypoint and waypoint not in route_with_waypoint:
                keep = False
                reason = "ไม่อยู่ในแนวเส้นทางหลักที่ระบบคำนวณใหม่ได้"
        else:
            if base_route and waypoint not in base_route and route_with_waypoint and waypoint not in route_with_waypoint:
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


def build_sensible_route(
    origin: str,
    destination: str,
    waypoints: list[str] | None = None,
    observed_provinces: list[str] | None = None,
) -> list[str]:
    waypoints = combine_ordered_provinces(waypoints or [])
    observed_provinces = combine_ordered_provinces(observed_provinces or [])

    backbone = infer_route_provinces_with_llm(
        origin,
        destination,
        waypoints,
        include_waypoints=bool(waypoints),
        force_include_waypoints=True,
    )
    if not backbone:
        backbone = combine_ordered_provinces(origin, waypoints, destination)

    if not observed_provinces:
        return backbone

    overlap = [province for province in observed_provinces if province in backbone]
    if len(overlap) < max(1, len(backbone) // 3):
        return backbone

    merged = list(backbone)
    for province in observed_provinces:
        if province not in merged and province not in {origin, destination}:
            merged.insert(-1, province)
    return merged


def get_route_provinces(origin: str, destination: str, waypoints: list[str] | None = None) -> tuple[list[str], float]:
    waypoints = combine_ordered_provinces(waypoints or [])

    orig_lat, orig_lng = geocode(origin)
    dest_lat, dest_lng = geocode(destination)
    if orig_lat is None or dest_lat is None:
        return build_sensible_route(origin, destination, waypoints), 0

    coords_str = f"{orig_lng},{orig_lat}"
    for waypoint in waypoints:
        wp_lat, wp_lng = geocode(waypoint)
        if wp_lat is not None and wp_lng is not None:
            coords_str += f";{wp_lng},{wp_lat}"
    coords_str += f";{dest_lng},{dest_lat}"

    try:
        response = http.get(
            f"{OSRM_URL}/{coords_str}",
            params={"overview": "full", "geometries": "geojson"},
            timeout=15,
        )
        data = response.json()
    except Exception:
        return build_sensible_route(origin, destination, waypoints), 0

    if data.get("code") != "Ok":
        return build_sensible_route(origin, destination, waypoints), 0

    coords = data["routes"][0]["geometry"]["coordinates"]
    distance_km = float(data["routes"][0]["distance"]) / 1000

    step = max(1, len(coords) // 8)
    sample_coords = coords[::step]
    found_provinces: list[str] = []

    for lng, lat in sample_coords:
        province = reverse_geocode_province(round(lat, 4), round(lng, 4))
        if province and province not in found_provinces:
            found_provinces.append(province)

    route_result = build_sensible_route(origin, destination, waypoints, observed_provinces=found_provinces)
    if len(route_result) < 3:
        route_result = combine_ordered_provinces(origin, found_provinces, waypoints, destination)

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
) -> str:
    lines: list[str] = []
    distance_text = f"ประมาณ {dist_km:.0f} กม." if dist_km > 0 else "ไม่สามารถประเมินระยะทางได้"

    lines.append(f"เส้นทาง: {origin} -> {destination} ({distance_text})")
    lines.append(f"จังหวัดที่ผ่าน: {' -> '.join(provinces) if provinces else '-'}")

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

    if q_type in {"realtime", "irrelevant"}:
        return (
            "ขออภัยค่ะ ระบบนี้ให้บริการเฉพาะข้อมูลสถิติอุบัติเหตุย้อนหลังเท่านั้น "
            "ไม่สามารถให้ข้อมูลเรียลไทม์หรือตอบคำถามที่ไม่เกี่ยวข้องได้ค่ะ",
            [],
        )

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

    if q_type == "route_analysis":
        origin, destination, waypoints = extract_origin_destination(question)
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
            )

            docs: list[dict[str, Any]] = []
            for province in provinces:
                docs.extend(fetch_docs_for_province(province, limit=6))
            return answer, docs[:40]

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

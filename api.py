#%%writefile api.py
import os
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ================== Константы / настройки ==================
SUPPORTED_FORMATS = {'.mp3', '.wav', '.ogg', '.webm'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

SAMPLE_RATE = 22050
N_MELS = 192
HOP_LENGTH = 256
N_FFT = 2048
POWER = 2.0
CROP_SECONDS = 12.0
CROP_FRAMES = int(round(CROP_SECONDS * SAMPLE_RATE / HOP_LENGTH))

# Добавляем параметры сегментации
SEGMENT_DURATION = 30  # seconds
SEGMENT_OVERLAP = 10   # seconds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = Path(os.getenv("WEIGHTS_PATH", "best_model_heavy_m192_h256_gru256x2.pth")).resolve()
LABELS_PATH = Path(os.getenv("LABELS_PATH", "labels.json")).resolve()

# ================== Модель (тяжёлая, как в обучении) ==================
class Large_CNN_RNN(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = N_MELS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.20),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.30),
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, n_mels, 512)
            out = self.conv(dummy)
            self.gru_input_size = out.size(1) * out.size(2)
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H)
        x = x.view(B, W, C * H)                 # (B, W, C*H)
        out, _ = self.gru(x)                    # (B, W, 512)
        out = out[:, -1, :]                     # (B, 512)
        return self.fc(out)                     # (B, n_classes)

# ================== Препроцесс под тяжёлую модель ==================
def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if y is None or len(y) == 0:
            return np.zeros(sr, dtype=np.float32)
        return y.astype(np.float32)
    except Exception:
        return np.zeros(sr, dtype=np.float32)

def to_mel_db(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT, power=POWER
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)  # (n_mels, T)
    return mel_db

def center_or_tta_crops(spec: np.ndarray, crop_frames: int = CROP_FRAMES, tta: int = 3) -> List[np.ndarray]:
    # Возвращает список кропов (по времени)
    T = spec.shape[1]
    if T <= crop_frames:
        pad = crop_frames - T
        if pad > 0:
            spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')
        return [spec[:, :crop_frames]]
    if tta <= 1:
        start = (T - crop_frames) // 2
        return [spec[:, start:start + crop_frames]]
    # Несколько равномерных кропов
    starts = np.linspace(0, T - crop_frames, num=tta, dtype=int)
    crops = [spec[:, s:s + crop_frames] for s in starts]
    return crops

def normalize(spec: np.ndarray) -> np.ndarray:
    m = spec.mean()
    s = spec.std() + 1e-6
    return (spec - m) / s

def crops_to_tensor(crops: List[np.ndarray]) -> torch.Tensor:
    # (N, 1, N_MELS, CROP_FRAMES)
    arr = np.stack([normalize(c) for c in crops], axis=0)
    arr = arr[:, None, :, :].astype(np.float32)
    return torch.from_numpy(arr)

# ================== Функции для сегментации ==================
def segment_audio(y: np.ndarray, sr: int = SAMPLE_RATE,
                  segment_duration: int = SEGMENT_DURATION,
                  overlap: int = SEGMENT_OVERLAP) -> List[np.ndarray]:
    """Разделение аудио на сегменты с перекрытием"""
    segment_samples = segment_duration * sr
    overlap_samples = overlap * sr
    step_samples = segment_samples - overlap_samples

    segments = []
    start = 0
    while start < len(y):
        end = min(start + segment_samples, len(y))
        segment = y[start:end]

        # Паддинг, если сегмент короче требуемой длины
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')

        segments.append(segment)
        start += step_samples

        # Если следующий сегмент будет полностью в пределах уже обработанного, останавливаемся
        if start >= len(y):
            break

    return segments

# ================== Загрузка labels ==================
def load_labels(labels_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        l2i = labels.get("label2idx", {})
        i2l_raw = labels.get("idx2label", {})
        i2l = {int(k): v for k, v in i2l_raw.items()}
        if not l2i or not i2l:
            raise RuntimeError("Некорректный labels.json")
        return l2i, i2l
    # Fallback (GTZAN, алфавитный порядок)
    classes = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    l2i = {c: i for i, c in enumerate(classes)}
    i2l = {i: c for c, i in l2i.items()}
    return l2i, i2l

# ================== Гибкая загрузка state_dict ==================
def clean_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        new_sd[k] = v
    return new_sd

def load_weights_strict(model: nn.Module, weights_path: Path):
    obj = torch.load(weights_path, map_location="cpu")
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    sd = clean_state_dict_keys(sd)
    model.load_state_dict(sd, strict=True)

# ================== Fallback: простая модель из model_utils (если нет тяжёлых весов) ==================
SimpleModel = None
simple_predict_fn = None
try:
    # Импортируем только при необходимости
    from model_utils import Simplified_CNN_RNN as SimpleModel, predict_audio_segmented , DEVICE as MU_DEVICE
except Exception:
    pass

# ================== Инициализация ==================
app = FastAPI(title="Music Genre Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

label2idx, idx2label = load_labels(LABELS_PATH)
n_classes = len(label2idx)

MODEL_TYPE = "heavy"
model: nn.Module
predict_fn = None
predict_segmented_fn = None

heavy_weights_exist = WEIGHTS_PATH.exists()
if heavy_weights_exist:
    try:
        model = Large_CNN_RNN(n_classes=n_classes, n_mels=N_MELS).to(DEVICE)
        load_weights_strict(model, WEIGHTS_PATH)
        model.eval()
        MODEL_TYPE = "heavy"

        def predict_heavy(path: str, top_k: int = 3) -> Tuple[str, float, List[Tuple[str, float]]]:
            y = load_audio(path, sr=SAMPLE_RATE)
            mel_db = to_mel_db(y, sr=SAMPLE_RATE)
            crops = center_or_tta_crops(mel_db, crop_frames=CROP_FRAMES, tta=3)
            x = crops_to_tensor(crops).to(DEVICE)
            with torch.no_grad():
                logits = model(x)  # (N, n_classes)
                probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
            top_idx = int(np.argmax(probs))
            top_label = idx2label.get(top_idx, "unknown")
            top_conf = float(probs[top_idx])
            # топ-k
            k = min(top_k, len(probs))
            topk_idx = np.argsort(probs)[::-1][:k]
            topk = [(idx2label.get(int(i), "unknown"), float(probs[i])) for i in topk_idx]
            return top_label, top_conf, topk

        def predict_heavy_segmented(path: str, top_k: int = 3) -> Tuple[str, float, List[Tuple[str, float]], List[Dict]]:
            """Предсказание жанра для сегментов аудио"""
            y = load_audio(path, sr=SAMPLE_RATE)
            segments = segment_audio(y, sr=SAMPLE_RATE, segment_duration=SEGMENT_DURATION, overlap=SEGMENT_OVERLAP)
            
            all_predictions = []
            segment_results = []
            
            for i, segment in enumerate(segments):
                # Преобразуем сегмент в мел-спектрограмму
                mel_db = to_mel_db(segment, sr=SAMPLE_RATE)
                crops = center_or_tta_crops(mel_db, crop_frames=CROP_FRAMES, tta=3)
                x = crops_to_tensor(crops).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(x)  # (N, n_classes)
                    probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
                
                # Получаем топ-1 предсказание для сегмента
                top_idx = int(np.argmax(probs))
                top_label = idx2label.get(top_idx, "unknown")
                top_conf = float(probs[top_idx])
                
                # Сохраняем результаты для сегмента
                segment_result = {
                    "segment": i+1,
                    "start_time": i * (SEGMENT_DURATION - SEGMENT_OVERLAP),
                    "end_time": min(i * (SEGMENT_DURATION - SEGMENT_OVERLAP) + SEGMENT_DURATION, len(y) / SAMPLE_RATE),
                    "genre": top_label,
                    "confidence": top_conf,
                    "all_probabilities": {idx2label.get(j, "unknown"): float(prob) for j, prob in enumerate(probs)}
                }
                segment_results.append(segment_result)
                all_predictions.append(probs)
            
            # Усредняем вероятности по всем сегментам
            avg_probs = np.mean(np.array(all_predictions), axis=0)
            top_idx = int(np.argmax(avg_probs))
            top_label = idx2label.get(top_idx, "unknown")
            top_conf = float(avg_probs[top_idx])
            
            # топ-k для усредненных вероятностей
            k = min(top_k, len(avg_probs))
            topk_idx = np.argsort(avg_probs)[::-1][:k]
            topk = [(idx2label.get(int(i), "unknown"), float(avg_probs[i])) for i in topk_idx]
            
            return top_label, top_conf, topk, segment_results

        predict_fn = predict_heavy
        predict_segmented_fn = predict_heavy_segmented
        print(f"Модель HEAVY загружена: {WEIGHTS_PATH.name} (classes={n_classes})")
    except Exception as e:
        print(f"Не удалось загрузить тяжёлые веса ({WEIGHTS_PATH}): {e}. Будет попытка fallback на простую модель.")
        heavy_weights_exist = False

if not heavy_weights_exist:
    if SimpleModel is None:
        raise RuntimeError(
            "Нет тяжёлых весов, и отсутствует model_utils.Simplified_CNN_RNN. "
            "Либо положите heavy чекпоинт, либо установите model_utils.py."
        )
    # Загрузим простую модель и, если есть, веса best_model.pth
    model = SimpleModel(n_classes=n_classes).to(DEVICE)
    simple_weights = Path(os.getenv("SIMPLE_WEIGHTS_PATH", "best_model.pth")).resolve()
    if simple_weights.exists():
        try:
            load_weights_strict(model, simple_weights)
            print(f"Модель SIMPLE загружена: {simple_weights.name}")
        except Exception as e:
            print(f"Не удалось загрузить {simple_weights.name}: {e}. Используется случайная инициализация.")
    else:
        print("best_model.pth не найден — используется случайная инициализация (качество будет низким).")

    model.eval()
    MODEL_TYPE = "simple"

    def predict_simple(path: str, top_k: int = 3) -> Tuple[str, float, List[Tuple[str, float]]]:
        # используем pipeline из model_utils
        genre, conf = predict_audio_segmented(path, model, label2idx, idx2label)
        # top-k недоступен из коробки — делаем повторный проход, чтобы получить softmax
        topk = [(genre, conf)]
        return genre, conf, topk

    # def predict_simple_segmented(path: str, top_k: int = 3) -> Tuple[str, float, List[Tuple[str, float]], List[Dict]]:
    #     """Предсказание жанра для сегментов аудио с простой моделью"""
    #     y = load_audio(path, sr=SAMPLE_RATE)
    #     segments = segment_audio(y, sr=SAMPLE_RATE, segment_duration=SEGMENT_DURATION, overlap=SEGMENT_OVERLAP)
        
    #     all_predictions = []
    #     segment_results = []
        
    #     for i, segment in enumerate(segments):
    #         # Сохраняем временный файл для сегмента
    #         segment_path = Path(f"temp_segment_{uuid.uuid4().hex}.wav")
    #         try:
    #             # Сохраняем сегмент во временный файл
    #             import soundfile as sf
    #             sf.write(str(segment_path), segment, SAMPLE_RATE)
                
    #             # Предсказание для сегмента
    #             genre, conf = predict_audio_segmented(str(segment_path), model, label2idx, idx2label)
                
    #             # Получаем вероятности для всех классов (приближенные)
    #             probs = np.zeros(n_classes)
    #             genre_idx = label2idx.get(genre, 0)
    #             probs[genre_idx] = conf
                
    #             # Сохраняем результаты для сегмента
    #             segment_result = {
    #                 "segment": i+1,
    #                 "start_time": i * (SEGMENT_DURATION - SEGMENT_OVERLAP),
    #                 "end_time": min(i * (SEGMENT_DURATION - SEGMENT_OVERLAP) + SEGMENT_DURATION, len(y) / SAMPLE_RATE),
    #                 "genre": genre,
    #                 "confidence": conf,
    #                 "all_probabilities": {idx2label.get(j, "unknown"): float(prob) for j, prob in enumerate(probs)}
    #             }
    #             segment_results.append(segment_result)
    #             all_predictions.append(probs)
    #         finally:
    #             # Удаляем временный файл
    #             if segment_path.exists():
    #                 segment_path.unlink()
        
        # # Усредняем вероятности по всем сегментам
        # avg_probs = np.mean(np.array(all_predictions), axis=0)
        # top_idx = int(np.argmax(avg_probs))
        # top_label = idx2label.get(top_idx, "unknown")
        # top_conf = float(avg_probs[top_idx])
        
        # # топ-k для усредненных вероятностей
        # k = min(top_k, len(avg_probs))
        # topk_idx = np.argsort(avg_probs)[::-1][:k]
        # topk = [(idx2label.get(int(i), "unknown"), float(avg_probs[i])) for i in topk_idx]
        
        # return top_label, top_conf, topk, segment_results

    predict_fn = predict_simple
    predict_segmented_fn = predict_simple_segmented
    print(f"Текущая модель: {MODEL_TYPE}, веса: {WEIGHTS_PATH if MODEL_TYPE == 'heavy' else os.getenv('SIMPLE_WEIGHTS_PATH', 'best_model.pth')}")

# ================== Роуты ==================
@app.get("/")
async def root():
    return {
        "message": "Music Genre Classification API is running",
        "device": str(DEVICE),
        "model_type": MODEL_TYPE,
        "classes": list(label2idx.keys()),
        "weights": str(WEIGHTS_PATH) if MODEL_TYPE == "heavy" else os.getenv("SIMPLE_WEIGHTS_PATH", "best_model.pth"),
    }

@app.get("/labels")
async def get_labels():
    return {"label2idx": label2idx, "idx2label": {int(k): v for k, v in idx2label.items()}}

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model_type": MODEL_TYPE}

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)):
    # Проверки
    name = file.filename or "file"
    if not any(name.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
        raise HTTPException(400, f"Неподдерживаемый формат: {name}")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, f"Файл слишком большой (> {MAX_FILE_SIZE} байт)")

    # Сохраняем временно (уникальное имя)
    suffix = Path(name).suffix or ".bin"
    temp_path = Path(f"temp_{uuid.uuid4().hex}{suffix}")
    try:
        temp_path.write_bytes(contents)
        label, conf, topk = predict_fn(str(temp_path), top_k=top_k)
        return {
            "model_type": MODEL_TYPE,
            "genre": label,
            "confidence": conf,
            "top_k": [{"label": l, "confidence": c} for l, c in topk],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка обработки: {e}")
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass

@app.post("/predict_segmented")
async def predict_segmented(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)):
    # Проверки
    name = file.filename or "file"
    if not any(name.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
        raise HTTPException(400, f"Неподдерживаемый формат: {name}")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, f"Файл слишком большой (> {MAX_FILE_SIZE} байт)")

    # Сохраняем временно (уникальное имя)
    suffix = Path(name).suffix or ".bin"
    temp_path = Path(f"temp_{uuid.uuid4().hex}{suffix}")
    try:
        temp_path.write_bytes(contents)
        label, conf, topk, segment_results = predict_segmented_fn(str(temp_path), top_k=top_k)
        return {
            "model_type": MODEL_TYPE,
            "overall_genre": label,
            "overall_confidence": conf,
            "top_k": [{"label": l, "confidence": c} for l, c in topk],
            "segments": segment_results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка обработки: {e}")
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"Запуск API на 0.0.0.0:{port} (DEVICE={DEVICE}, MODEL_TYPE={MODEL_TYPE})")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
import os
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

# Устройство
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Аудио/фичи (те же каналы, что у вас: 40+128+12+7+6=193)
SAMPLE_RATE = 22050
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
MAX_TIME = 800   # временная длина (пад/обрезка)
DURATION = 30    # сек
SEGMENT_DURATION = 30  # seconds
SEGMENT_OVERLAP = 10   # seconds

def load_audio(path: str, sr: int = SAMPLE_RATE, duration: Optional[int] = DURATION) -> np.ndarray:
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
        if y is None or len(y) == 0:
            return np.zeros(sr * (duration or DURATION), dtype=np.float32)
        return y.astype(np.float32)
    except Exception:
        return np.zeros(sr * (duration or DURATION), dtype=np.float32)

def extract_enhanced_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)

    # Mel
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    feat = np.vstack([mfcc, mel_db, chroma, spectral_contrast, tonnetz]).astype(np.float32)
    return feat

def preprocess_for_model(feat: np.ndarray, max_time: int = MAX_TIME) -> torch.Tensor:
    # feat shape: (channels=193, time)
    # Пад/обрезка по времени
    if feat.shape[1] < max_time:
        pad = max_time - feat.shape[1]
        feat = np.pad(feat, ((0, 0), (0, pad)), mode='constant')
    else:
        feat = feat[:, :max_time]

    # Нормализация по sample
    m = feat.mean()
    s = feat.std() + 1e-6
    feat = (feat - m) / s

    # В модель: (B, C_in=1, H=193, W=max_time)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return x

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

class Simplified_CNN_RNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),
        )

        # Автоматический расчёт размеров для GRU
        with torch.no_grad():
            dummy = torch.randn(1, 1, 193, MAX_TIME)
            out = self.conv(dummy)
            # out: (B, C, H, W)
            self.gru_input_size = out.size(1) * out.size(2)  # C * H
            # последовательность по ширине
            # дальше permute (B, W, C*H)
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                # (B, C, H, W)
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2)       # (B, W, C, H)
        x = x.contiguous().view(B, W, C * H)  # (B, W, C*H)
        x, _ = self.gru(x)              # (B, W, 256)
        x = x[:, -1, :]                 # (B, 256)
        x = self.fc(x)                  # (B, n_classes)
        return x

#def predict_audio(path: str, model: nn.Module, label2idx: Dict[str, int], idx2label: Dict[int, str]) -> Tuple[str, float]:
    #y = load_audio(path, sr=SAMPLE_RATE, duration=DURATION)
    #feat = extract_enhanced_features(y, sr=SAMPLE_RATE)
    #x = preprocess_for_model(feat, max_time=MAX_TIME).to(DEVICE)
    # model.eval()
    # with torch.no_grad():
    #     logits = model(x)
    #     probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # pred_idx = int(np.argmax(probs))
    # genre = idx2label.get(pred_idx, "unknown")
    # conf = float(probs[pred_idx])
    # return genre, conf

def predict_audio_segmented(path: str, model: nn.Module, label2idx: Dict[str, int], idx2label: Dict[int, str], 
                           top_k: int = 3) -> Tuple[str, float, List[Tuple[str, float]], List[Dict]]:
    """Предсказание жанра для сегментов аудио с простой моделью"""
    y = load_audio(path, sr=SAMPLE_RATE, duration=None)  # Загружаем весь файл
    segments = segment_audio(y, sr=SAMPLE_RATE, segment_duration=SEGMENT_DURATION, overlap=SEGMENT_OVERLAP)
    
    all_predictions = []
    segment_results = []
    n_classes = len(label2idx)
    
    for i, segment in enumerate(segments):
        # Извлекаем признаки для сегмента
        feat = extract_enhanced_features(segment, sr=SAMPLE_RATE)
        x = preprocess_for_model(feat, max_time=MAX_TIME).to(DEVICE)
        
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Получаем топ-1 предсказание для сегмента
        pred_idx = int(np.argmax(probs))
        genre = idx2label.get(pred_idx, "unknown")
        conf = float(probs[pred_idx])
        
        # Сохраняем результаты для сегмента
        segment_result = {
            "segment": i+1,
            "start_time": i * (SEGMENT_DURATION - SEGMENT_OVERLAP),
            "end_time": min(i * (SEGMENT_DURATION - SEGMENT_OVERLAP) + SEGMENT_DURATION, len(y) / SAMPLE_RATE),
            "genre": genre,
            "confidence": conf,
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
    
    with open("last_segmented_prediction.txt", "a", encoding="utf-8") as f:
        f.write("file: {}\n".format(Path(path).name))
        f.write("top_label: {}\n".format(top_label))
        f.write("top_confidence: {}\n".format(top_conf))
        f.write("top_k: {}\n".format(topk))
        f.write("segments: {}\n".format(segment_results))
        f.write("all predictions: {}\n".format(all_predictions))
    return top_label, top_conf, topk, segment_results
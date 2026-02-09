import torch
import librosa

AUDIO_LABEL_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad"
}

TEXT_TO_EMOTION = {
    "POSITIVE": "happy",
    "NEGATIVE": "sad",
    "NEUTRAL": "neutral"
}

HIGH_CONF = 0.55


def extract_sentence_audio(audio_path, segments, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    chunks = []

    for seg in segments:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)

        chunks.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
            "audio": y[start:end]
        })

    return chunks


def audio_emotion(audio_chunk, feature_extractor, audio_model, sr=16000):
    if len(audio_chunk) < sr * 0.3:
        return "neutral", 0.0

    inputs = feature_extractor(
        audio_chunk,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = audio_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    pred_id = torch.argmax(probs, dim=-1).item()
    conf = probs.max().item()

    emotion = AUDIO_LABEL_MAP[audio_model.config.id2label[pred_id]]
    return emotion, round(conf, 3)


def text_emotion(text, text_model):
    result = text_model(text)[0]
    return TEXT_TO_EMOTION[result["label"]], round(result["score"], 3)


def fuse_emotion(audio_em, audio_conf, text_em, text_conf):
    if audio_em == text_em:
        return audio_em, round((audio_conf + text_conf) / 2, 3)

    if audio_conf >= HIGH_CONF:
        return audio_em, audio_conf

    if audio_em == "angry":
        return audio_em, audio_conf

    return text_em, text_conf


def run_pipeline(audio_path, models):
    feature_extractor, audio_model, whisper_model, text_model = models

    whisper_result = whisper_model.transcribe(
        audio_path, word_timestamps=True
    )

    sentence_chunks = extract_sentence_audio(
        audio_path, whisper_result["segments"]
    )

    timeline = []

    for s in sentence_chunks:
        aud_em, aud_conf = audio_emotion(
            s["audio"], feature_extractor, audio_model
        )
        txt_em, txt_conf = text_emotion(
            s["text"], text_model
        )

        final_em, final_conf = fuse_emotion(
            aud_em, aud_conf, txt_em, txt_conf
        )

        timeline.append({
            "start_time": s["start"],
            "end_time": s["end"],
            "text": s["text"],
            "emotion": final_em,
            "confidence": final_conf
        })

    return timeline

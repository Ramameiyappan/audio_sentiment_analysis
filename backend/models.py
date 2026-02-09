import whisper
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    pipeline
)

def load_models():
    AUDIO_MODEL = "superb/wav2vec2-base-superb-er"

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL)
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_MODEL)
    audio_model.eval()

    whisper_model = whisper.load_model("base")

    text_sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return feature_extractor, audio_model, whisper_model, text_sentiment_model
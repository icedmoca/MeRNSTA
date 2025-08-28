import os
import io
import tempfile
import pytest
from fastapi.testclient import TestClient
from api.routes.memory import router
from storage.spacy_extractor import SpacyTripletExtractor
from storage.memory_log import MemoryLog

client = TestClient(router)

@pytest.fixture(scope="module")
def extractor():
    return SpacyTripletExtractor()

@pytest.fixture(scope="module")
def memory_log():
    return MemoryLog()

def test_image_caption_and_triplet_extraction(extractor):
    # Use a small red square as a mock image
    from PIL import Image
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    triplets = extractor.extract_triplets(media_type="image", media_data=image_bytes)
    assert triplets, "No triplets extracted from image"
    t = triplets[0]
    assert t.subject == "user"
    assert t.predicate == "described"
    assert "red" in t.object or "photo" in t.object

def test_audio_transcription_and_triplet_extraction(extractor):
    # Use a short silent WAV as mock audio (simulate "I like blue")
    import numpy as np
    import soundfile as sf
    import tempfile
    # Generate a 1-second silent audio (simulate for test)
    data = np.zeros(16000)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, data, 16000)
    with open(tmp.name, "rb") as f:
        audio_bytes = f.read()
    # This will not transcribe "I like blue" but tests the pipeline
    triplets = extractor.extract_triplets(media_type="audio", media_data=audio_bytes)
    assert isinstance(triplets, list)
    # Clean up
    os.remove(tmp.name)

def test_store_and_search_image_fact(memory_log, extractor):
    # Store an image fact
    from PIL import Image
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    triplets = extractor.extract_triplets(media_type="image", media_data=image_bytes)
    t = triplets[0]
    tup = (t.subject, t.predicate, t.object, getattr(t, "confidence", 1.0), getattr(t, "context", None), "image", image_bytes)
    ids, msgs = memory_log.store_triplets([tup])
    assert ids, "Image fact not stored"
    # Search for "red car" in images
    results = memory_log.semantic_search("red car", media_type="image", topk=3)
    assert any(r["media_type"] == "image" for r in results)

def test_store_and_search_audio_fact(memory_log, extractor):
    # Store an audio fact (mock)
    import numpy as np
    import soundfile as sf
    import tempfile
    data = np.zeros(16000)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, data, 16000)
    with open(tmp.name, "rb") as f:
        audio_bytes = f.read()
    triplets = extractor.extract_triplets(media_type="audio", media_data=audio_bytes)
    if triplets:
        t = triplets[0]
        tup = (t.subject, t.predicate, t.object, getattr(t, "confidence", 1.0), getattr(t, "context", None), "audio", audio_bytes)
        ids, msgs = memory_log.store_triplets([tup])
        assert ids, "Audio fact not stored"
        # Search for "like blue" in audio
        results = memory_log.semantic_search("like blue", media_type="audio", topk=3)
        assert any(r["media_type"] == "audio" for r in results)
    os.remove(tmp.name)

def test_api_upload_and_search_image():
    # Use a small red square as a mock image
    from PIL import Image
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    files = {"file": ("test.png", image_bytes, "image/png")}
    data = {"media_type": "image", "description": "red square"}
    response = client.post("/memory/upload_media", files=files, data=data)
    assert response.status_code == 200
    # Now search
    resp = client.get("/memory/search_multimodal", params={"query": "red car", "media_type": "image"})
    assert resp.status_code == 200
    assert "results" in resp.json()

def test_api_upload_and_search_audio():
    # Use a short silent WAV as mock audio
    import numpy as np
    import soundfile as sf
    import tempfile
    data = np.zeros(16000)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, data, 16000)
    with open(tmp.name, "rb") as f:
        audio_bytes = f.read()
    files = {"file": ("test.wav", audio_bytes, "audio/wav")}
    data = {"media_type": "audio", "description": "silent audio"}
    response = client.post("/memory/upload_media", files=files, data=data)
    assert response.status_code == 200
    # Now search
    resp = client.get("/memory/search_multimodal", params={"query": "like blue", "media_type": "audio"})
    assert resp.status_code == 200
    assert "results" in resp.json()

def test_no_sentence_transformers():
    import sys
    assert 'sentence_transformers' not in sys.modules, 'sentence_transformers should not be imported' 
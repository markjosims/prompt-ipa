from app import *
from transformers import WhisperForConditionalGeneration, WhisperProcessor

TEST_WAV = 'test_data/sample_biling.wav'

def test_pipeline_object():
    """
    Assert that pipeline correctly loads Whisper model and processor
    """
    pipe = ContextualBiasingASR_Pipeline(model_path=WHISPER_TINY, processor_path=WHISPER_TINY)
    assert isinstance(pipe.model, WhisperForConditionalGeneration)
    assert isinstance(pipe.processor, WhisperProcessor)

def test_pipeline_output():
    """
    Assert that pipeline returns text transcription for input audio
    """
    pipe = ContextualBiasingASR_Pipeline(model_path=WHISPER_TINY, processor_path=WHISPER_TINY)
    wav = load_and_resample(TEST_WAV)
    chunks = chunk_audio(wav, chunk_len_s=10, return_timestamps=False)
    out = pipe(chunks=chunks)
    assert type(out) is dict
    assert 'text' in out
    assert type(out['text']) is str

def test_prompt_pipeline():
    """
    Assert that pipeline returns text transcription for input audio
    """
    pipe = ContextualBiasingASR_Pipeline(model_path=WHISPER_TINY, processor_path=WHISPER_TINY)
    wav = load_and_resample(TEST_WAV)
    chunks = chunk_audio(wav, chunk_len_s=10, return_timestamps=False)
    prompts = ["foo bar baz"] * len(chunks)
    out = pipe(chunks=chunks, prompts=prompts)
    assert type(out) is dict
    assert 'text' in out
    assert type(out['text']) is str
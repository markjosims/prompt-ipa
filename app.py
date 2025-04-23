from typing import Optional, Union, List
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    AutomaticSpeechRecognitionPipeline
)
import torch
import torchaudio
from tqdm import tqdm
import numpy as np

# constants
WHISPER_TINY = 'openai/whisper-tiny'
WHISPER_SMALL = 'openai/whisper-small'
WHISPER_MEDIUM = 'openai/whisper-medium'
WHISPER_LARGE_V2 = 'openai/whisper-large-v2'
WHISPER_LARGE_v3 = 'openai/whisper-large-v3'

SAMPLE_RATE = 16_000

# -------------- #
# pipeline class #
# -------------- #

class ContextualBiasingASR_Pipeline():
    """
    Similar functionality to `transformers.AutomaticSpeechRecognitionPipeline` without chunking.
    Instead, expects a list of audio chunks as input alongside a list of prompts.
    """
    def __init__(self, model_path: str, processor_path: Optional[str]):
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.processor = WhisperProcessor.from_pretrained(processor_path)

    def __call__(
            self,
            chunks: List[np.ndarray],
            prompts: List[str] = None,
        ):
        """
        - `chunks`: List of numpy arrays containing samples for each chunk of audio
        - `prompts`: [Optional] list of strs indicating prompt for each chunk

        Chunks audio then runs `self.model.generate` on each chunk within the audio.
        If `prompts` is passed, then prepends each to the context for each chunk.
        """
        
        # process audio
        inputs = self.processor(chunks, return_tensors='pt')

        # infer on chunks iteratively
        if prompts is None:
            prompts = ['' for _ in chunks]
        output = {'text': ''}
        with torch.no_grad():
            for i, input_features in tqdm(enumerate(inputs['input_features'])):
                prompt = prompts[i]
                prompt_ids = self.processor.get_prompt_ids(prompt, return_tensors='pt')
                output_ids = self.model.generate(
                    input_features=input_features.unsqueeze(dim=0),
                    prompt_ids = prompt_ids
                )
                output_text = self.processor.decode(output_ids.squeeze())
                output['text'] = ' '.join([output['text'].strip(), output_text.strip()])
                if i<len(inputs)-1:
                    prompts[i+1]=' '.join([prompts[i+1].strip(), output_text.strip()])
        return output




# ------------- #
# audio helpers #
# ------------- #

def load_and_resample(
        fp: Union[str, List[str]],
        sr: int = SAMPLE_RATE,
        to_mono: bool = True,
        flatten: bool = True,
        to_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
    f"""
    Load a wavfile at filepath `fp` into a torch tensor.
    Resample to `sr` (default {SAMPLE_RATE}).
    If `to_mono` is passed, convert to mono by dropping the second channel.
    If `flatten` is also passed, squeeze.
    """
    if type(fp) is list:
        return [load_and_resample(sub_fp, sr=sr, to_mono=to_mono, flatten=flatten) for sub_fp in fp]
    wav_orig, sr_orig = torchaudio.load(fp)
    wav = torchaudio.functional.resample(wav_orig, sr_orig, sr)
    if to_mono and len(wav.shape)==2:
        wav=wav[:1,:]
        if flatten:
            wav=wav.squeeze()
    elif flatten:
        raise ValueError("Cannot flatten wav unless converting to mono!")
    if to_numpy:
        return wav.numpy()
    return wav

def get_frame(
        audio: torch.Tensor,
        frame_start: int,
        frame_end: int,
        sample_rate: int = SAMPLE_RATE,
        return_timestamps: bool = False,        
):
    f"""
    Slice a frame from the audio tensor indicated by the sample indices `frame_start` and `frame_end`.
    If `return_timestamps=True`, instead return a dict with keys `start_s` (start time in seconds),
    `end_s` (end time in seconds) and `samples` (tensor of wav samples for the given frame).
    Pass `sample_rate` to override the default sample rate of {SAMPLE_RATE}.
    """
    if return_timestamps:
        frame_start_s = frame_start/sample_rate
        frame_end_s = frame_end/sample_rate
        return {
            'start_s': frame_start_s,
            'end_s': frame_end_s,
            'samples': audio[frame_start:frame_end]
        }
    return audio[frame_start:frame_end]

def get_sliding_window(
        audio: torch.Tensor,
        frame_len_s: float,
        frameshift_s: float,
        sample_rate: int = SAMPLE_RATE,
        return_timestamps: bool = False,
    ):
    f"""
    Split audio tensor into a list of tensors, each corresponding to a frame of length `framelength_s`
    staggered by `frameshift_s`. If `return_timestamps=True`, return a list of dictionaries with keys `start_s`
    (start time in seconds), `end_s` (end time in seconds) and `samples` (tensor of wav samples for the given frame).
    Pass `sample_rate` to override the default sample rate of {SAMPLE_RATE}.
    """
    if len(audio)==0:
        return []
    framelength_samples = int(frame_len_s * sample_rate)
    frameshift_samples = int(frameshift_s * sample_rate)
    
    frame_start = 0
    frame_end = framelength_samples
    windows = []
    while frame_end<len(audio):
        frame = get_frame(audio, frame_start, frame_end, sample_rate, return_timestamps)
        windows.append(frame)
        frame_start+=frameshift_samples
        frame_end+=frameshift_samples
    # append last truncated frame
    frame = get_frame(audio, frame_start, len(audio), sample_rate, return_timestamps)
    windows.append(frame)
    
    return windows

def chunk_audio(
        audio: torch.Tensor,
        chunk_len_s: float,
        sample_rate: int = SAMPLE_RATE,
        return_timestamps: bool = False,
):
    """
    Calls `get_sliding_window` with `frameshift_s=frame_len_s_s=chunk_len_s`. 
    """
    return get_sliding_window(
        audio=audio,
        frame_len_s=chunk_len_s,
        frameshift_s=chunk_len_s,
        sample_rate=sample_rate,
        return_timestamps=return_timestamps,
    )

def main():
    ...

if __name__ == "__main__":
    main()
from typing import List, Union

from librosa import resample
import torch
import numpy as np
from transformers import SpeechT5FeatureExtractor, SpeechT5HifiGan


class LogMelTransform:
    PRETRAINED_VOCODER_NAME = 'microsoft/speecht5_hifigan'
    PRETRAINED_FEATURE_EXTRACTOR_NAME = 'microsoft/speecht5_tts'

    def __init__(self):
        self.feature_extractor = SpeechT5FeatureExtractor.from_pretrained(self.PRETRAINED_FEATURE_EXTRACTOR_NAME)
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.PRETRAINED_VOCODER_NAME)

    @property
    def sampling_rate(self):
        return self.feature_extractor.sampling_rate

    def transform(
        self,
        waveforms: Union[np.ndarray, List[np.ndarray]],
        sampling_rate: int = 16000
    ):
        if not isinstance(waveforms, list):
            waveforms = [waveforms]

        if sampling_rate != self.sampling_rate:
            for i in range(len(waveforms)):
                waveforms[i] = resample(waveforms[i], orig_sr=sampling_rate, target_sr=self.sampling_rate)
        
        return self.feature_extractor(
            audio_target=waveforms,
            sampling_rate=self.sampling_rate,
            return_tensors='pt'
        )

    @torch.inference_mode()
    def inv_transform(self, specs: torch.FloatTensor):
        waveforms = self.vocoder(specs)
        return waveforms

import numpy as np
import torch
import torch.nn as nn
import glob
from audio_utils import get_audio_chunks
from display_utils import map_w2v_to_quadrant
from evaluation import ccc
from scipy.io.wavfile import read
from librosa import load
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


# define constants
SAMPLING_RATE = 16000
IS_JL = True

# load model from hub
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)

# load all wav files in a directory into a list
files = glob.glob("data/jl/*.wav")
signals = [[] for i in range(len(files))]

# Create lists for storing annotations
true_val = []
true_aro = []

pred_val = []
pred_aro = []

for i, file in enumerate(files):
    current_signal = load(file, sr=SAMPLING_RATE)
    current_signal = [current_signal[0]]
    signals[i].append(current_signal)
    chunks = get_audio_chunks(
        signal=current_signal[0], frame_size=200, sampling_rate=16000, is_jl=IS_JL)
    for chunk in chunks:
        chunk = [[np.array(chunk, dtype=np.float32)]]
        result = process_func(chunk, SAMPLING_RATE)
        result[0][0], result[0][2], result[0][1] = map_w2v_to_quadrant(
            result[0][0], result[0][2], result[0][1])
        pred_aro.append(result[0][0])
        pred_val.append(result[0][2])
    break

true_val = [-0.6, -0.6, -0.62, -0.62, -0.58, -0.58, -0.6, -0.6]
true_aro = [0.86, 0.84, 0.84, 0.85, 0.8, 0.81, 0.85, 0.85]

true_val = np.array(true_val)
true_aro = np.array(true_aro)
pred_val = np.array(pred_val)
pred_aro = np.array(pred_aro)

ccc_val = ccc(pred_val, true_val)
ccc_aro = ccc(pred_aro, true_aro)
print("Pred val:", pred_val)
print("Pred aro:", pred_aro)
print(f'CCC for arousal: {ccc_aro} \nCCC for valence: {ccc_val}')


# results = [[] for i in range(len(files))]

# # process loaded signals
# for j, signal in enumerate(signals):
#     results[j].append(process_func(signal, SAMPLING_RATE))
#     print(results[j][0])


# print(process_func(signal_dummy, sampling_rate))
#  Arousal    dominance valence
# [[0.5460759 0.6062269 0.4043165]]

# process_func(signals, SAMPLING_RATE, embeddings=True)
# Pooled hidden states of last transformer layer
# [[-0.00752167  0.0065819  -0.00746339 ...  0.00663631  0.00848747
#   0.00599209]]

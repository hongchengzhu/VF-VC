from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoProcessor, AutoModelForPreTraining
from datasets import load_dataset
import soundfile as sf
import torch
import soundfile as sf

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# tokenize
# input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1
input = torch.tensor(sf.read('../wavs/LJ001-0001.wav')[0], dtype=torch.float32).to('cpu').unsqueeze(0)
# retrieve logits
logits = model(input).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)

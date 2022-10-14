import pytorch_lightning as pl
import torch.nn as nn
from transformers import ElectraModel, AutoTokenizer
import torch
from util.emotion_class import LABELS

device = "cuda" if torch.cuda.is_available() else "cpu"


class KOTEtagger(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.electra = ElectraModel.from_pretrained("beomi/KcELECTRA-base").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
        self.classifier = nn.Linear(self.electra.config.hidden_size, 44).to(device)

    def forward(self, text: str):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        ).to(device)
        output = self.electra(encoding["input_ids"], attention_mask=encoding["attention_mask"])
        output = output.last_hidden_state[:, 0, :]
        output = self.classifier(output)
        output = torch.sigmoid(output)
        torch.cuda.empty_cache()

        return output


trained_model = KOTEtagger()
trained_model.load_state_dict(torch.load("./checkpoint/kote_pytorch_lightning.bin"))  # <All keys matched successfully>라는 결과가 나오는지 확인!

preds = trained_model(
    """재미있어요! 재미는 확실히 있는데 뭐랄까... 너무 정신 없달까...ㅋㅋ"""
)[0]

for l, p in zip(LABELS, preds):
    print(device)
    if p > 0.4:
        print(f"{l}: {p}")


def top_k_emotions(texts: list, threshold: float, k: int):
    if not 0 <= threshold <= 1:
        raise ValueError("theshold must be a float b/w 0 ~ 1.")
    results = {}
    for text in texts:
        cur_result = {}
        for out in pipe(text)[0]:
            if out["score"] > threshold:
                cur_result[out["label"]] = round(out["score"], 2)
        cur_result = sorted(cur_result.items(), key=lambda x: x[1], reverse=True)
        results[text] = cur_result[:k]

    return results
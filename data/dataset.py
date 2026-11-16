import os
import unicodedata

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

TARGET_H = 32


class LineDataset(Dataset):
    """Pary {base}.png / {base}.txt w jednym katalogu.

    Kolejność filtrów: najpierw alfabet (znak poza charsetem), potem CTC
    (labelka dłuższa niż W/4 kroków czasowych po resize).
    """

    def __init__(self, root, charset):
        self.root = root
        self.charset = charset
        self.allowed = set(charset.stoi)
        self.skipped_ctc = 0
        self.skipped_charset = 0
        self.skipped_names = []
        ids = []
        for f in sorted(os.listdir(root)):
            if not f.endswith(".png"):
                continue
            base = os.path.splitext(f)[0]
            txt = os.path.join(root, base + ".txt")
            if not os.path.exists(txt):
                continue
            with open(txt, encoding="utf-8") as fh:
                # jawne UTF-8 + NFC — przed strip i encode
                text = unicodedata.normalize("NFC", fh.read()).strip()

            outside = set(text) - self.allowed
            if outside:
                self.skipped_charset += 1
                self.skipped_names.append(base)
                continue

            with Image.open(os.path.join(root, f)) as img:   # czyta tylko nagłówek
                w = max(1, round(img.width * TARGET_H / img.height))
            if len(text) > w // 4:
                self.skipped_ctc += 1
                continue
            ids.append(base)
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        base = self.ids[i]
        img = Image.open(os.path.join(self.root, base + ".png")).convert("L")

        with open(os.path.join(self.root, base + ".txt"), encoding="utf-8") as f:
            # jawne UTF-8 + NFC — przed strip i encode
            text = unicodedata.normalize("NFC", f.read()).strip()

        w = max(1, round(img.width * TARGET_H / img.height))
        img = img.resize((w, TARGET_H))
        x = TF.to_tensor(img)  # (1, 32, W), float 0–1

        y = torch.tensor(self.charset.encode(text), dtype=torch.long)
        return x, y


def collate(batch):
    xs, ys = zip(*batch, strict=True)     # każdy item: dokładnie (x, y)
    # prawdziwe T per próbka — LICZONE PRZED paddingiem (W/4 po CNN)
    input_lens = torch.tensor([x.shape[2] // 4 for x in xs])
    W = max(x.shape[2] for x in xs)
    xs = torch.stack([F.pad(x, (0, W - x.shape[2])) for x in xs])  # (B,1,H,W)
    y_lens = torch.tensor([len(y) for y in ys])
    return xs, torch.cat(ys), input_lens, y_lens

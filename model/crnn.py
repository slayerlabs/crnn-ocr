"""CRNN dla OCR linii tekstu: CNN (kompresja szerokości 4x) -> BiLSTM -> FC.

Kontrakt:
  wejście:  (B, 1, 32, W)  grayscale, wysokość 32, dowolna szerokość
  wyjście:  (B, T, C)      T = W/4 kroków czasowych, C = liczba klas (blank=0)
"""
import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int, lstm_hidden: int = 256, lstm_layers: int = 2):
        super().__init__()

        # --- CNN: bloki conv3x3 -> BN -> ReLU -> pool.
        # W pionie: 32 -> 16 -> 8 -> 1  (poolingi spłaszczają wysokość do 1).
        # W poziomie: W -> W/2 -> W/4  (dwa halvingi; pool bloku 3 tylko pionowy).
        # Kontrakt T = W/4 (nie W/8): więcej kroków czasowych łatwiej
        # spełnia warunek CTC (labelka <= T; W/8 odrzucałoby ~50% tego datasetu).
        self.cnn = nn.Sequential(
            # blok 1: 32xW -> 16x(W/2)
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # blok 2: 16x(W/2) -> 8x(W/4)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # blok 3: 8x(W/4) -> 1x(W/4)  (pool tylko w pionie, szerokość bez zmian)
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((8, 1)),          # pion 8->1, poziom zostaje
            # blok 4: konwolucje na "kolumnach czasowych", bez poolingu
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        # Mapowanie kanałów -> cech per kolumna
        # po CNN: (B, 256, 1, W/4) -> squeeze wymiaru H -> (B, 256, W/4)

        # --- RNN: BiLSTM per kolumna czasowa
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,   # (B, T, features) — ważne!
        )
        # UWAGA: nie przekazujemy hidden state między batchami (patrz forward)

        # --- Klasyfikacja per krok czasowy
        self.fc = nn.Linear(2 * lstm_hidden, num_classes)  # 2x bo bidirectional

    def forward(self, x):                  # x: (B, 1, 32, W)
        f = self.cnn(x)                    # (B, 256, 1, W/4)
        f = f.squeeze(2).permute(0, 2, 1)  # (B, T=W/4, 256)  — batch_first
        f, _ = self.lstm(f)                # (B, T, 2*hidden)
        logits = self.fc(f)                # (B, T, C)
        return logits


if __name__ == "__main__":
    # mini smoke test kształtów — uruchamiany bez danych
    m = CRNN(num_classes=105)
    # szerokości: podzielne przez 4, niepodzielne przez 4, duża "brzydka"
    for W in (128, 250, 777):
        x = torch.randn(2, 1, 32, W) # Tensor z losowym kształtem obrazka
        y = m(x)
        T = W // 4
        assert y.shape == (2, T, 105), f"W={W}: {y.shape}"
        print(f"W={W:>4}  ->  logits {tuple(y.shape)}  (T={T})  OK")
    print("CRNN shape check OK")

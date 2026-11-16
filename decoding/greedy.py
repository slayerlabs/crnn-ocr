def greedy_decode(logits, charset):
    """logits: (B, T, C) → lista tekstów. Blank=0, usuwanie powtórzeń."""
    out = []
    for seq in logits.argmax(-1):          # (T,)
        chars, prev = [], None
        for i in seq.tolist():
            if i != prev and i != 0:      # skip blank i powtórki
                chars.append(charset.itos[i])
            prev = i
        out.append("".join(chars))
    return out

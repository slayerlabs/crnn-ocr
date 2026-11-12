"""Beam search CTC (prefix beam) — kontrakt identyczny jak greedy_decode:

    beam_decode(logits (B, T, C), charset, beam_size=10) -> lista tekstów

Blank=0. Stan beama: etykieta + masa ścieżek zakończonych blankiem (p_b)
i nie-blankiem (p_nb); powtórka znaku zwija się do prefixu, duplikat po
blanku wyduża. Po każdym kroku normalizacja przez masę najlepszego beama
(ranking bez zmian, zero underflow) i cut do beam_size.
"""

from collections import defaultdict


def _beam_search_one(probs, beam_size):
    """probs: lista wierszy (prawdopodobieństwa klas per krok) -> krotka id."""
    beams = {(): (1.0, 0.0)}                 # prefix -> (p_blank, p_nonblank)
    for row in probs:
        n_classes = len(row)
        p_blank = row[0]
        nxt = defaultdict(lambda: [0.0, 0.0])
        for prefix, (p_b, p_nb) in beams.items():
            mass = p_b + p_nb
            entry = nxt[prefix]
            entry[0] += mass * p_blank       # blank: prefix bez zmian
            last = prefix[-1] if prefix else None
            for c in range(1, n_classes):
                p_c = row[c]
                if c == last:
                    entry[1] += p_nb * p_c              # powtórka po znaku
                    nxt[prefix + (c,)][1] += p_b * p_c  # po blanku -> nowa litera
                else:
                    nxt[prefix + (c,)][1] += mass * p_c
        # ranking po masie całkowitej; normalizacja najlepszym beamem
        top = sorted(nxt.items(),
                     key=lambda kv: -(kv[1][0] + kv[1][1]))[:beam_size]
        mx = top[0][1][0] + top[0][1][1]
        beams = {pref: (e[0] / mx, e[1] / mx) for pref, e in top}
    return max(beams, key=lambda pref: beams[pref][0] + beams[pref][1])


def beam_decode(logits, charset, beam_size=10):
    """logits: (B, T, C) -> lista tekstów; hipoteza = najmasywniejszy prefix."""
    probs = logits.log_softmax(-1).exp()     # wewn. konwersja jak w greedy argmax
    out = []
    for b in range(logits.size(0)):
        ids = _beam_search_one(probs[b].tolist(), beam_size)
        out.append("".join(charset.itos[i] for i in ids))
    return out

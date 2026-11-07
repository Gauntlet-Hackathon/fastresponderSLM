"""Deterministic parsing of NEED_DURATION spans into hours, and attaching them to a need kind.

The model only finds the words ("for 30 hours", "desde ayer", "2 days"); turning them into a number
is code, so "2 days" is always exactly 48 and the engine's `> 24 h` rule is never at the mercy of a
learned regression.
"""

from __future__ import annotations

import re
import unicodedata

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "couple of": 2, "couple": 2, "few": 3, "several": 3,
    "un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "doce": 12, "quince": 15, "veinte": 20,
    "treinta": 30, "cuarenta": 40, "par de": 2, "un par de": 2, "varios": 3, "varias": 3,
}
# Exact unit spellings (no prefix matching: "dialysis" is not "dia", "minimum" is not "min").
UNIT_HOURS = {
    **dict.fromkeys(("minute", "minutes", "min", "mins", "minuto", "minutos"), 1 / 60),
    **dict.fromkeys(("hour", "hours", "hr", "hrs", "h", "hora", "horas"), 1.0),
    **dict.fromkeys(("day", "days", "dia", "dias", "night", "nights", "noche", "noches"), 24.0),
    **dict.fromkeys(("week", "weeks", "semana", "semanas"), 168.0),
}
PHRASES = [  # checked first, on accent-stripped lowercase text
    (r"\b(since|from) yesterday\b|\bdesde ayer\b|\ball day yesterday\b", 24.0),
    (r"\bsince last night\b|\bdesde anoche\b|\bovernight\b|\btoda la noche\b", 12.0),
    (r"\bsince (this )?morning\b|\bdesde (esta )?(la )?manana\b", 8.0),
    (r"\b(since|for) (the )?(whole|entire) day\b|\btodo el dia\b", 12.0),
    (r"\bsince (the storm|the hurricane|the flood)\b|\bdesde (la tormenta|el huracan)\b", None),
]


def _alt(words) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


# <number> [+|-] <unit>: "30 hours", "36hrs", "two days", "a couple of days", "dos dias".
NUM_UNIT = re.compile(
    rf"\b(\d+(?:[.,]\d+)*|{_alt(NUMBER_WORDS)})\s*(?:\+|-)?\s*({_alt(UNIT_HOURS)})\b"
)
LONE_UNIT = re.compile(rf"\b({_alt(u for u in UNIT_HOURS if len(u) > 3)})\b")
NEED_WORDS = {
    "water": ("water", "h2o", "agua", "drink", "beber", "thirst", "sed"),
    "food": ("food", "eat", "comida", "comer", "hungry", "hambre", "alimento", "formula", "milk",
             "leche"),
    "meds": ("med", "insulin", "insulina", "prescription", "pill", "medicin", "medicamento",
             "pastilla", "receta", "oxygen", "oxigeno", "inhaler", "inhalador"),
}
NEED_WINDOW = 40  # a duration attaches to a need keyword at most this many characters away


def plain(text: str) -> str:
    """Lowercase and strip accents, one output character per input character, so indices into
    the result are indices into the original text."""
    out = []
    for c in text:
        low = c.lower()
        low = low if len(low) == 1 else c
        base = "".join(x for x in unicodedata.normalize("NFKD", low)
                       if not unicodedata.combining(x))
        out.append(base if len(base) == 1 else low)
    return "".join(out)


def _number(token: str) -> float:
    if token[0].isdigit():
        if re.fullmatch(r"\d{1,3}(,\d{3})+", token):  # "1,000": thousands separator
            return float(token.replace(",", ""))
        return float(token.replace(",", "."))  # "1,5": decimal comma
    return float(NUMBER_WORDS[token])


def parse_hours(span_text: str) -> float | None:
    """Hours described by a duration phrase, or None if it cannot be read."""
    t = plain(span_text)
    for pattern, hours in PHRASES:
        if re.search(pattern, t):
            return hours
    found = [_number(n) * UNIT_HOURS[u] for n, u in NUM_UNIT.findall(t)]
    if found:
        return round(max(found), 2)
    lone = LONE_UNIT.search(t)  # "all day", "toda la semana"
    return UNIT_HOURS[lone.group(1)] if lone else None


def nearest_need(text: str, start: int, end: int, kinds: list[str]) -> str | None:
    """The need kind (among `kinds`) whose keyword is closest to the span [start, end), if one is
    within NEED_WINDOW characters; otherwise None (the duration is about something else)."""
    p = plain(text)
    best, best_dist = None, None
    for kind in kinds:
        for word in NEED_WORDS[kind]:
            for m in re.finditer(re.escape(word), p):
                dist = start - m.end() if m.end() <= start else max(0, m.start() - end)
                if dist <= NEED_WINDOW and (best_dist is None or dist < best_dist):
                    best, best_dist = kind, dist
    return best

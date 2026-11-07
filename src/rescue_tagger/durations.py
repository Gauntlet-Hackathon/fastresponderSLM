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
    "thirty": 30, "forty": 40, "fifty": 50, "couple": 2, "few": 3, "several": 3,
    "un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "doce": 12, "quince": 15, "veinte": 20,
    "treinta": 30, "cuarenta": 40, "par": 2, "varios": 3, "varias": 3,
}
UNIT_HOURS = {
    "minute": 1 / 60, "min": 1 / 60, "minuto": 1 / 60,
    "hour": 1, "hr": 1, "h": 1, "hora": 1,
    "day": 24, "dia": 24, "night": 24, "noche": 24,
    "week": 168, "semana": 168,
}
PHRASES = [  # checked in order, on accent-stripped lowercase text
    (r"\b(since|from) yesterday\b|\bdesde ayer\b|\ball day yesterday\b", 24.0),
    (r"\bsince last night\b|\bdesde anoche\b|\bovernight\b|\btoda la noche\b", 12.0),
    (r"\bsince (this )?morning\b|\bdesde (esta )?(la )?manana\b", 8.0),
    (r"\b(since|for) (the )?(whole|entire) day\b|\btodo el dia\b", 12.0),
    (r"\bsince (the storm|the hurricane|the flood)\b|\bdesde (la tormenta|el huracan)\b", None),
]
NUM_UNIT = re.compile(r"(\d+(?:[.,]\d+)?|[a-z]+)\s*-?\s*(?:\+\s*)?([a-z]+)")
NEED_WORDS = {
    "water": ("water", "h2o", "agua", "drink", "beber", "thirst", "sed"),
    "food": ("food", "eat", "comida", "comer", "hungry", "hambre", "alimento", "formula", "milk",
             "leche"),
    "meds": ("med", "insulin", "insulina", "prescription", "pill", "medicin", "medicamento",
             "pastilla", "receta", "oxygen", "oxigeno", "inhaler", "inhalador"),
}


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _unit_hours(word: str) -> float | None:
    for stem in sorted(UNIT_HOURS, key=len, reverse=True):
        if word == stem or word == stem + "s" or word == stem + "es" or (
            len(stem) > 2 and word.startswith(stem)
        ):
            return UNIT_HOURS[stem]
    return None


def parse_hours(span_text: str) -> float | None:
    """Hours described by a duration phrase, or None if it cannot be read."""
    t = _plain(span_text)
    for pattern, hours in PHRASES:
        if re.search(pattern, t):
            return hours
    best = None
    for num, unit in NUM_UNIT.findall(t):
        if num.replace(",", ".").replace(".", "", 1).isdigit():
            value = float(num.replace(",", "."))
        elif num in NUMBER_WORDS:
            value = float(NUMBER_WORDS[num])
        else:
            continue
        per = _unit_hours(unit)
        if per is not None:
            hours = value * per
            best = hours if best is None else max(best, hours)
    if best is None:
        # "a day", "un dia": a unit word on its own means one of it.
        for word in re.findall(r"[a-z]+", t):
            per = _unit_hours(word)
            if per is not None and word not in ("h", "min"):
                return per
    return round(best, 2) if best is not None else None


def nearest_need(text: str, start: int, end: int, kinds: list[str]) -> str | None:
    """The need kind (among `kinds`) whose keyword is closest to the span [start, end)."""
    if not kinds:
        return None
    if len(kinds) == 1:
        return kinds[0]
    plain = _plain(text)
    best, best_dist = None, None
    for kind in kinds:
        for word in NEED_WORDS[kind]:
            for m in re.finditer(re.escape(word), plain):
                dist = start - m.end() if m.end() <= start else max(0, m.start() - end)
                if best_dist is None or dist < best_dist:
                    best, best_dist = kind, dist
    return best

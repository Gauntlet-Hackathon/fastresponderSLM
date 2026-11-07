"""Offline P1 safety net: send a request to a human when its words suggest a P1 hazard or a
vulnerable person that the model did not report, or when the model itself is unsure.

  from rescue_tagger.safety import review_flags
  flags = review_flags(text, payload)   # [] = no human review needed

The model can be confidently blind to a hazard that is implied rather than named ("trapped in
the attic" means rising water; "my son's ventilator" means a child). A confidence rule cannot
catch that, so this adds a second, independent signal: an English / Spanish / Vietnamese word
list per rubric kind, written from the rubric's definitions (never from evaluation rows), with
common misspellings and texting forms. It never changes the model's payload; it only says "a
person should look at this", which is the rubric's answer to doubt. It runs offline in well
under a millisecond.

Flags:
  unsure_request   is_request confidence in [0.25, 0.75)
  near:<kind>      the model gave <kind> a confidence in [0.2, 0.5): seen but not reported
  cue:<kind>       the text has a cue word for <kind> and the model gave it < 0.2
A message the model is confidently sure is not a request (>= 0.75) is not flagged by cue words,
so news and offers of help that mention water or roofs don't flood the review queue.
"""

from __future__ import annotations

import re
from typing import Any

ON = 0.5            # a kind counts as reported at this confidence (as in product_eval)
NEAR = 0.2          # a kind between NEAR and ON is "seen but not reported"
UNSURE = (0.25, 0.75)

# Patterns match at a word start, case-insensitively; a trailing \b marks a whole word. Vietnamese
# patterns accept the unaccented spelling too (people often type without diacritics).
_EN_STUCK = (
    r"trapp?ed", r"trap\b", r"stuck", r"stck\b", r"stranded", r"can'?t (?:get|go) (?:out|anywhere)",
    r"cannot (?:get|go) out", r"cant get out", r"can'?t leave", r"unable to (?:leave|get out)",
    r"no way out", r"(?:boxed|cut|closed) (?:in|off)", r"pinned", r"locked in", r"blocked in",
    r"attic", r"crawl ?space", r"(?:car|truck|vehicle|suv|van) (?:is |got )?(?:stuck|stalled|"
    r"swept|floating|underwater|submerged)", r"underpass", r"(?:swept|washed) away",
    r"surrounded by (?:water|flood)", r"water all around", r"island(?:ed)?\b",
)
_ES_STUCK = (
    r"atrapad", r"atrapa", r"atorad", r"atascad", r"varad", r"encerrad", r"aislad",
    r"no (?:podemos|puedo|pueden|puede) salir", r"sin salida", r"rodead[oa]s? de agua",
    r"[áa]tico", r"desv[áa]n", r"(?:carro|coche|troca|camioneta) (?:atorad|atascad|flotando|"
    r"arrastrad|bajo el agua)", r"arrastrad",
)
_VI_STUCK = (
    r"m[ắa]c k[ẹe]t", r"b[ịi] k[ẹe]t", r"k[ẹe]t\b", r"k[ẹe]t l[ạa]i", r"k[ẹe]t c[ứu]ng",
    r"kh[ôo]ng (?:ra|tho[áa]t|đi|di|ra ngo[àa]i) (?:ra |ra ngo[àa]i )?(?:đ|d)[ưu][ợo]c",
    r"ko (?:ra|tho[áa]t|đi|di) (?:đ|d)[ưu][ợo]c", r"g[áa]c x[ée]p", r"g[áa]c m[áa]i",
    r"b[ịi] c[ôo] l[ậa]p", r"c[ôo] l[ậa]p", r"b[ịi] cu[ốo]n tr[ôo]i", r"cu[ốo]n tr[ôo]i",
)
_EN_WATER = (
    r"(?:water|watr|wtr|flood ?water|flood\w*) (?:is |are |keeps |still )?(?:rising|risin|coming|"
    r"up\b|to\b|at\b|in\b|into|inside|over|past|getting|filling|pouring|chest|waist|knee|neck)",
    r"rising (?:water|fast)", r"flood(?:ed|ing)? (?:house|home|apartment|street|car|room)",
    r"(?:house|home|apartment|street|car|room|first floor|1st floor|downstairs) (?:is |are )?"
    r"(?:flood|under ?water|submerged)", r"under ?water", r"submerged", r"chest[- ]deep",
    r"waist[- ]deep", r"knee[- ]deep", r"neck[- ]deep", r"\d+ ?(?:ft|feet|foot|in|inches) of water",
    r"water (?:level|keeps)", r"attic", r"swimming", r"drowning", r"sinking", r"levee",
    r"(?:dam|bayou|river|creek) (?:broke|overflow|breach|is over)", r"overflow",
)
_ES_WATER = (
    r"(?:el )?agua (?:ya |est[áa] |sigue |va )?(?:sube|subiendo|hasta|entrando|entr[óo]|llega|"
    r"lleg[óo]|al pecho|a la cintura|a las rodillas|al cuello|por encima)", r"inund",
    r"se (?:est[áa] )?(?:llenando|metiendo) (?:de )?agua", r"bajo el agua", r"crecida",
    r"desbord", r"ahog", r"hundi", r"pies de agua", r"metros de agua", r"[áa]tico",
)
_VI_WATER = (
    r"n[ưu][ớo]c (?:đ|d)?(?:ang )?(?:l[êe]n|d[âa]ng|v[àa]o|t[ớo]i|ng[ậa]p|cao|tr[àa]n|s[âa]u|"
    r"ngang|qu[áa])", r"ng[ậa]p", r"l[ũu]\b", r"l[ũu] l[ụu]t", r"l[ụu]t", r"tr[àa]n v[àa]o",
    r"đu[ốo]i n[ưu][ớo]c", r"chìm", r"g[áa]c x[ée]p",
)
CUES: dict[str, tuple[str, ...]] = {
    "trapped": _EN_STUCK + _ES_STUCK + _VI_STUCK,
    "rising_water": _EN_WATER + _ES_WATER + _VI_WATER,
    "on_roof": (
        r"roofs?\b", r"rooftop", r"on (?:the|our|my) ro+f", r"rof\b", r"ruf\b", r"top of (?:the|our) "
        r"(?:house|car|truck|building)", r"second story roof", r"porch roof", r"shingles",
        r"techo", r"azotea", r"tejado", r"encima de la casa", r"arriba de la casa",
        r"m[áa]i nh[àa]", r"n[óo]c nh[àa]", r"tr[êe]n m[áa]i", r"tr[êe]n n[óo]c", r"s[âa]n th[ượu]ng",
    ),
    "fire": (
        r"fire", r"smoke", r"smoking\b", r"smell(?:s|ing)? (?:gas|smoke|burning)", r"gas leak",
        r"burn(?:ing|t|ed)?\b", r"flames?", r"spark(?:s|ing)", r"electrical fire", r"explosion",
        r"exploded", r"on fire",
        r"fuego", r"humo", r"incendi", r"llamas", r"quem", r"huele a (?:gas|quemado|humo)",
        r"fuga de gas", r"chispas", r"explosi[óo]n", r"explot",
        r"ch[áa]y\b", r"đ[áa]m ch[áa]y", r"kh[óo]i", r"l[ửu]a", r"r[òo] r[ỉi] gas", r"m[ùu]i gas",
        r"m[ùu]i kh[ée]t", r"n[ổo]\b", r"ch[ậa]p đi[ệe]n",
    ),
    "injury": (
        r"hurt", r"injur", r"wound", r"broke(?:n)? (?:arm|leg|bone|hip|ankle|wrist|back|neck|rib)",
        r"broken", r"fractur", r"bleed", r"blood", r"cut (?:bad|badly|open|deep)", r"gash",
        r"unconscious", r"passed out", r"not breathing", r"can'?t breathe", r"hit (?:his|her|my|their) head",
        r"head injury", r"concuss", r"electrocut", r"shocked", r"burned (?:his|her|my|their)",
        r"in pain", r"pain\b", r"fell\b", r"fall(?:en)?\b", r"bitten", r"snake ?bite",
        r"herid", r"lastimad", r"lesionad", r"sangr", r"(?:brazo|pierna|hueso|cadera|tobillo) rot",
        r"fractura", r"inconsciente", r"desmay", r"no respira", r"golpe en la cabeza", r"electrocut",
        r"dolor", r"se cay[óo]", r"mordid",
        r"b[ịi] th[ưu][ơo]ng", r"th[ưu][ơo]ng t[íi]ch", r"ch[ảa]y m[áa]u", r"g[ãa]y\b", r"b[ấa]t t[ỉi]nh",
        r"ng[ấa]t", r"kh[ôo]ng th[ởo] (?:đ|d)[ưu][ợo]c", r"b[ịi] (?:đ|d)i[ệe]n gi[ậa]t", r"b[ịi] ng[ãa]",
        r"(?:đ|d)au", r"r[ắa]n c[ắa]n",
    ),
    "medical_device_no_power": (
        r"oxygen", r"o2\b", r"ventilator", r"vent\b", r"respirator", r"dialysis", r"c-?pap",
        r"bi-?pap", r"concentrator", r"nebuli[sz]er", r"feeding (?:tube|pump)", r"suction machine",
        r"insulin pump", r"power(?:ed)? wheelchair", r"electric wheelchair", r"life support",
        r"machine (?:to breathe|for breathing)", r"breathing machine",
        r"ox[íi]geno", r"ventilador", r"respirador", r"di[áa]lisis", r"concentrador",
        r"nebulizador", r"m[áa]quina (?:para respirar|de respirar|de oxigeno)", r"bomba de insulina",
        r"silla de ruedas el[ée]ctrica",
        r"m[áa]y th[ởo]", r"m[áa]y oxy", r"b[ìi]nh oxy", r"oxy\b", r"ô xy", r"l[ọo]c th[ậa]n",
        r"ch[ạa]y th[ậa]n", r"m[áa]y (?:t[ạa]o|h[úu]t) (?:oxy|đ[ờo]m)", r"xe l[ăa]n đi[ệe]n",
    ),
    "child": (
        r"son\b", r"sons\b", r"daughters?", r"kids?\b", r"kiddos?", r"child", r"children", r"chlid",
        r"baby", r"babies", r"bby\b", r"infant", r"toddler", r"newborn", r"little (?:one|ones|boy|"
        r"girl)", r"boy\b", r"girl\b", r"grandkids?", r"grand(?:son|daughter|child)", r"nephew",
        r"niece", r"(?:[1-9]|1[0-7]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"[- ](?:yo|y/o|year[- ]old|month[- ]old|years? old|months? old)", r"\d+ ?(?:mo|month)s?\b",
        r"ni[ñn][oa]s?", r"nin[oa]s?\b", r"hij[oa]s?", r"beb[ée]s?", r"bebito", r"bebita",
        r"chiquit", r"peque[ñn]", r"criatura", r"nieto", r"nieta", r"sobrin", r"mis chamacos",
        r"chamac", r"escuincl", r"menor(?:es)?\b",
        r"con trai", r"con g[áa]i", r"con t[ôo]i", r"con nh[ỏo]", r"em b[ée]", r"b[ée] (?:trai|g[áa]i)",
        r"tr[ẻe] (?:em|con|nh[ỏo]|s[ơo] sinh)", r"ch[áa]u (?:n[ộo]i|ngo[ạa]i|nh[ỏo])", r"đứa (?:b[ée]|nh[ỏo])",
        r"(?:\d+|m[ộo]t|hai|ba) (?:đứa )?(?:con|ch[áa]u)\b", r"con\b",
    ),
    "elderly": (
        r"elderly", r"eldery", r"old (?:man|men|woman|women|lady|ladies|people|peopel|ppl|couple|"
        r"folks|parents|neighbor)", r"older (?:man|woman|lady|people|neighbor|couple|parents)",
        r"aged\b", r"seniors?\b", r"grand(?:ma|pa|mother|father|parents?|mom|dad)", r"granny",
        r"nana\b", r"papaw", r"mamaw", r"meemaw", r"great[- ]grand",
        r"(?:[6-9]\d|1[01]\d)[- ]?(?:yo|y/o|yrs?|years?[- ]old|year[- ]old)", r"in (?:his|her|their) "
        r"(?:70|80|90)s", r"nursing home", r"assisted living", r"retirement home", r"dementia",
        r"alzheimer",
        r"abuel", r"ancian", r"viejit", r"viej[oa]s?\b", r"persona(?:s)? mayor", r"tercera edad",
        r"(?:se[ñn]or|se[ñn]ora) mayor", r"(?:[6-9]\d|1[01]\d) a[ñn]os", r"asilo", r"demencia",
        r"ng[ưu][ờo]i gi[àa]", r"ng[ưu][ờo]i l[ớo]n tu[ổo]i", r"cao tu[ổo]i", r"[ôo]ng b[àa]",
        r"c[ụu]\b", r"c[ụu] gi[àa]", r"b[àa] n[ộo]i", r"b[àa] ngo[ạa]i", r"[ôo]ng n[ộo]i",
        r"[ôo]ng ngo[ạa]i", r"b[àa] c[ụu]", r"[ôo]ng c[ụu]", r"(?:[6-9]\d|1[01]\d) tu[ổo]i",
        r"vi[ệe]n d[ưu][ỡo]ng l[ãa]o", r"b[ốo] m[ẹe] gi[àa]",
    ),
    "disabled": (
        r"wheel ?chair", r"wheelchiar", r"disab", r"handicap", r"can'?t walk", r"cannot walk",
        r"unable to walk", r"bed ?ridden", r"bed ?bound", r"paraly", r"quadripleg", r"parapleg",
        r"amputee", r"walker\b", r"crutches", r"blind\b", r"deaf\b", r"mobility", r"special needs",
        r"autis", r"down syndrome", r"cerebral palsy", r"stroke\b", r"immobile",
        r"silla de ruedas", r"discapacit", r"minusv[áa]lid", r"no (?:puede|pueden) caminar",
        r"no camina", r"en cama", r"encamad", r"paralizad", r"par[áa]lisis", r"ciego", r"ciega",
        r"sord[oa]", r"andadera", r"muletas", r"autis", r"necesidades especiales",
        r"xe l[ăa]n", r"khuy[ếe]t t[ậa]t", r"t[àa]n t[ậa]t", r"li[ệe]t\b", r"b[ạa]i li[ệe]t",
        r"kh[ôo]ng (?:đ|d)i (?:đ|d)[ưu][ợo]c", r"n[ằa]m li[ệe]t", r"m[ùu]\b", r"(?:đ|d)i[ếe]c",
        r"t[ựu] k[ỷi]", r"(?:đ|d)[ộo]t qu[ỵy]",
    ),
    "medical": (
        r"diabet", r"insulin", r"dialysis", r"oxygen", r"chemo", r"cancer", r"seizure", r"epilep",
        r"heart (?:condition|problem|attack|failure|patient)", r"asthma", r"copd", r"inhaler",
        r"medication", r"meds\b", r"medicine", r"prescription", r"surgery", r"post[- ]op",
        r"sick\b", r"ill\b", r"fever", r"infection", r"high blood pressure", r"hypertension",
        r"kidney", r"transplant", r"hospice", r"blood sugar", r"insulin", r"epipen", r"allerg",
        r"patient", r"stroke\b",
        r"diab[ée]t", r"insulina", r"quimio", r"c[áa]ncer", r"convulsi", r"epilep", r"asma",
        r"coraz[óo]n", r"medicina", r"medicamento", r"pastillas", r"receta", r"cirug[íi]a",
        r"operad[oa]", r"enferm", r"fiebre", r"infecci[óo]n", r"presi[óo]n alta", r"ri[ñn][óo]n",
        r"az[úu]car", r"paciente",
        r"ti[ểe]u (?:đ|d)[ưu][ờo]ng", r"(?:đ|d)[ưu][ờo]ng huy[ếe]t", r"thu[ốo]c\b", r"ung th[ưu]",
        r"h[óo]a tr[ịi]", r"(?:đ|d)[ộo]ng kinh", r"co gi[ậa]t", r"tim\b", r"b[ệe]nh tim",
        r"hen suy[ễe]n", r"huy[ếe]t [áa]p", r"m[ổo]\b", r"ph[ẫa]u thu[ậa]t", r"b[ệe]nh\b",
        r"[ốo]m\b", r"s[ốo]t\b", r"th[ậa]n\b", r"b[ệe]nh nh[âa]n",
    ),
    "pregnant": (
        r"pregnant", r"preggo", r"pregant", r"expecting", r"in labor", r"labor pains",
        r"contractions", r"water broke", r"due (?:date|any day|soon)", r"weeks along",
        r"(?:[1-4]\d|[5-9]) (?:weeks|months) pregnant",
        r"embarazad", r"embarasad", r"encinta", r"de parto", r"contracciones", r"se le rompi[óo] la "
        r"fuente", r"rompi[óo] fuente", r"dar a luz", r"esperando (?:un )?beb[ée]",
        r"mang thai", r"c[óo] b[ầa]u", r"c[óo] thai", r"b[ầa]u\b", r"chuy[ểe]n d[ạa]", r"s[ắa]p sinh",
        r"v[ỡo] [ốo]i", r"s[ắa]p (?:đ|d)[ẻe]", r"(?:đ|d)au (?:đ|d)[ẻe]",
    ),
}
_RX = {k: re.compile(r"(?<!\w)(?:" + "|".join(v) + r")", re.IGNORECASE) for k, v in CUES.items()}


def cue_kinds(text: str) -> set[str]:
    """Kinds whose cue words appear in the text."""
    return {k for k, rx in _RX.items() if rx.search(text)}


def _conf(payload: dict[str, Any], kind: str) -> float:
    for item in payload.get("hazards", []) + payload.get("vulnerable", []):
        if item["kind"] == kind:
            return float(item.get("confidence", 1.0))
    return 0.0


def review_flags(text: str, payload: dict[str, Any]) -> list[str]:
    """Reasons a person should review this message; [] when none."""
    flags: list[str] = []
    req = payload["is_request"]
    c = float(req["confidence"])
    if UNSURE[0] <= c < UNSURE[1]:
        flags.append("unsure_request")
    if not req["value"] and c >= UNSURE[1]:
        return flags          # confidently not a request: cue words alone do not flag it
    for kind in sorted(CUES):
        k_conf = _conf(payload, kind)
        if NEAR <= k_conf < ON:
            flags.append(f"near:{kind}")
        elif k_conf < NEAR and _RX[kind].search(text):
            flags.append(f"cue:{kind}")
    return flags

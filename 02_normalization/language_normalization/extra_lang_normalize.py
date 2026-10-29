#!/usr/bin/env python3

"""
batch_second_stage_language_cleanup.py

Applies second-stage typed language cleanup across multiple WorldBench views.

Supports:
    1. dataset-level list-like language fields (e.g. language_list)
    2. exploded single-language fields (e.g. language)

Outputs:
    - cleaned copy of each CSV
    - optional audit summary CSV per file
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FILE_CONFIGS = [
    {
        "name": "worldbench_base_ground_truth",
        "input_csv": "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/worldbench_base_ground_truth.csv",
        "language_column": "language_list",
        "mode": "list",
    },
    {
        "name": "worldbench_lang_tierA",
        "input_csv": "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/worldbench_lang_tierA.csv",
        "language_column": "language_list",
        "mode": "list",
    },
    {
        "name": "worldbench_lang_tierB_excluded",
        "input_csv": "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/worldbench_lang_tierB_excluded.csv",
        "language_column": "language_list",
        "mode": "list",
    },
    {
        "name": "worldbench_lang_tierC_naive",
        "input_csv": "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/worldbench_lang_tierC_naive.csv",
        "language_column": "language_list",
        "mode": "list",
    },
    {
        "name": "worldbench_lang_tierA_expanded",
        "input_csv": "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/worldbench_lang_tierA_expanded.csv",
        "language_column": "language",
        "mode": "single",
    },
    {
        "name": "worldbench_lang_tierC_expanded",
        "input_csv": "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/worldbench_lang_tierC_expanded.csv",
        "language_column": "language",
        "mode": "single",
    },
    {
        "name": "worldbench_country_only_expanded",
        "input_csv": "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/worldbench_country_only_expanded.csv",
        "language_column": "language_list",   # change to languages_in_dataset if needed
        "mode": "list",
    },
    {
        "name": "worldbench_country_task_expanded",
        "input_csv": "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/worldbench_country_task_expanded.csv",
        "language_column": "language_list",   # change to languages_in_dataset if needed
        "mode": "list",
    },
]

OUTPUT_DIR = "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2"


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_basic_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = text.replace("–", "-").replace("—", "-")
    return text.strip()


def normalize_match_key(text: str) -> str:
    text = normalize_basic_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ============================================================
# MAPS / TAXONOMY
# Expand over time as you audit outputs.
# ============================================================

DIRECT_MAP: Dict[str, str] = {
    "english": "English",
    "anglais": "English",
    "english (uk)": "English",
    "north american english": "English",
    "mainstream u.s. english": "English",
    "south african english": "English",
    "standardized english": "English",
    "indian english": "English",

    "spanish": "Spanish",
    "español": "Spanish",
    "espagnol": "Spanish",

    "french": "French",
    "français": "French",

    "german": "German",
    "deutsch": "German",
    "standard german": "German",
    "simple german": "German",

    "portuguese": "Portuguese",
    "português": "Portuguese",

    "italian": "Italian",
    "italiano": "Italian",
    "italien": "Italian",

    "bulgarian": "Bulgarian",
    "български": "Bulgarian",

    "czech": "Czech",
    "čeština": "Czech",

    "lithuanian": "Lithuanian",
    "lietuvių": "Lithuanian",

    "greek": "Greek",
    "ελληνικά": "Greek",
    "modern greek": "Greek",

    "dutch": "Dutch",
    "nederlands": "Dutch",

    "romanian": "Romanian",
    "română": "Romanian",

    "slovak": "Slovak",
    "slovenčina": "Slovak",

    "slovenian": "Slovenian",
    "slovenščina": "Slovenian",

    "croatian": "Croatian",
    "hrvatski": "Croatian",

    "latvian": "Latvian",
    "latviešu": "Latvian",

    "polish": "Polish",
    "polski": "Polish",

    "swedish": "Swedish",
    "svenska": "Swedish",

    "finnish": "Finnish",
    "suomi": "Finnish",

    "danish": "Danish",
    "dansk": "Danish",

    "hungarian": "Hungarian",
    "magyar": "Hungarian",

    "chinese": "Chinese",
    "中文": "Chinese",
    "standard chinese": "Chinese",
    "written chinese": "Chinese",
    "simplified chinese": "Chinese",
    "traditional chinese": "Chinese",
    "chinese (simplified)": "Chinese",
    "chinese (traditional)": "Chinese",
    "chinese simplified": "Chinese",
    "chinese (mandarin)": "Chinese",

    "urdu": "Urdu",
    "odia": "Odia",
    "oriya": "Odia",
    "odiya": "Odia",
    "telegu": "Telugu",
    "telugu": "Telugu",
    "afaan oromo": "Oromo",
    "afan oromo": "Oromo",
    "oromo": "Oromo",
    "tigrigna": "Tigrinya",
    "tigrinya": "Tigrinya",
    "kinyrwanda": "Kinyarwanda",
    "kinyarwanda": "Kinyarwanda",
    "kiswahili": "Swahili",
    "swahili": "Swahili",
    "zulu": "Zulu",
    "isizulu": "Zulu",
    "xhosa": "Xhosa",
    "isixhosa": "Xhosa",
    "bahasa indonesia": "Indonesian",
    "indonesian": "Indonesian",
    "tagalog": "Tagalog",
    "filipino": "Filipino",
    "maori": "Māori",
    "māori": "Māori",
    "guarani": "Guaraní",
    "guaraní": "Guaraní",
}

DIRECT_MAP.update({
    # High-frequency canonical languages
    "arabic": "Arabic",
    "hindi": "Hindi",
    "japanese": "Japanese",
    "bengali": "Bengali",
    "tamil": "Tamil",
    "russian": "Russian",
    "korean": "Korean",
    "malayalam": "Malayalam",
    "marathi": "Marathi",
    "turkish": "Turkish",
    "kannada": "Kannada",
    "persian": "Persian",
    "vietnamese": "Vietnamese",
    "gujarati": "Gujarati",
    "ukrainian": "Ukrainian",
    "hebrew": "Hebrew",
    "punjabi": "Punjabi",
    "assamese": "Assamese",
    "icelandic": "Icelandic",
    "sanskrit": "Sanskrit",
    "nepali": "Nepali",
    "basque": "Basque",
    "norwegian": "Norwegian",
    "catalan": "Catalan",
    "amharic": "Amharic",
    "javanese": "Javanese",
    "latin": "Latin",
    "estonian": "Estonian",
    "serbian": "Serbian",
    "manipuri": "Manipuri",
    "hausa": "Hausa",
    "kazakh": "Kazakh",
    "sinhala": "Sinhala",
    "irish": "Irish",
    "konkani": "Konkani",
    "tibetan": "Tibetan",
    "sindhi": "Sindhi",
    "uyghur": "Uyghur",
    "maltese": "Maltese",
    "yoruba": "Yoruba",
    "galician": "Galician",
    "bhojpuri": "Bhojpuri",
    "kurdish": "Kurdish",
    "maithili": "Maithili",
    "luxembourgish": "Luxembourgish",
    "afrikaans": "Afrikaans",
    "somali": "Somali",
    "uzbek": "Uzbek",
    "welsh": "Welsh",
    "faroese": "Faroese",
    "malay": "Malay",
    "albanian": "Albanian",
    "nahuatl": "Nahuatl",
    "occitan": "Occitan",
    "kashmiri": "Kashmiri",
    "setswana": "Tswana",
    "minangkabau": "Minangkabau",
    "dogri": "Dogri",
    "santali": "Santali",
    "meitei": "Manipuri",
    "armenian": "Armenian",
    "magahi": "Magahi",
    "yiddish": "Yiddish",
    "balinese": "Balinese",
    "sesotho": "Southern Sotho",
    "hmong": "Hmong",
    "buginese": "Buginese",
    "georgian": "Georgian",
    "cebuano": "Cebuano",
    "macedonian": "Macedonian",
    "pashto": "Pashto",
    "esperanto": "Esperanto",
    "amazigh": "Amazigh",
    "breton": "Breton",
    "burmese": "Burmese",
    "aymara": "Aymara",
    "bambara": "Bambara",
    "haitian creole": "Haitian Creole",
    "belarusian": "Belarusian",
    "scottish gaelic": "Scottish Gaelic",
    "wolof": "Wolof",
    "asturian": "Asturian",
    "azerbaijani": "Azerbaijani",
    "inuktitut": "Inuktitut",
    "kyrgyz": "Kyrgyz",
    "kokborok": "Kokborok",
    "tatar": "Tatar",
    "tetun": "Tetum",
    "lingala": "Lingala",
    "bribri": "Bribri",
    "erzya": "Erzya",
    "corsican": "Corsican",
    "mixtec": "Mixtec",
    "rajasthani": "Rajasthani",
    "malagasy": "Malagasy",
    "chatino": "Chatino",
    "frisian": "Frisian",
    "upper sorbian": "Upper Sorbian",
    "khoekhoe": "Khoekhoe",
    "gorontalo": "Gorontalo",
    "puyuma": "Puyuma",
    "aromanian": "Aromanian",
    "kalaallisut": "Kalaallisut",
    "gondi": "Gondi",
    "khmer": "Khmer",
    "dzongkha": "Dzongkha",
    "zhuang": "Zhuang",
    "kirundi": "Kirundi",
    "aramaic": "Aramaic",
    "fijian": "Fijian",
    "papiamento": "Papiamento",
    "samoan": "Samoan",
    "tongan": "Tongan",
    "romansh": "Romansh",
    "aragonese": "Aragonese",
    "aranese": "Aranese",
    "tuvan": "Tuvan",
    "newar": "Newar",
    "ojibwe": "Ojibwe",
    "mazatec": "Mazatec",
    "otomi": "Otomi",
    "kurukh": "Kurukh",
    "ladin": "Ladin",
    "hawaiian": "Hawaiian",
    "yaqui": "Yaqui",
    "turkmen": "Turkmen",
    "shona": "Shona",
    "fante": "Fante",
    "mende": "Mende",
    "kikongo": "Kikongo",
    "sango": "Sango",
    "balochi": "Balochi",
    "chavacano": "Chavacano",
    "kapampangan": "Kapampangan",
    "ilokano": "Ilokano",
    "waray": "Waray",
    "tausug": "Tausug",
    "pulaar": "Pulaar",
    "sereer": "Sereer",
    "saraiki": "Saraiki",
    "khinalug": "Khinalug",
    "manchu": "Manchu",
    "wayuunaiki": "Wayuunaiki",
    "awajun": "Awajun",
    "amuzgo": "Amuzgo",
    "zarma": "Zarma",
    "abaza": "Abaza",
    "moksha": "Moksha",
    "udmurt": "Udmurt",
    "cree": "Cree",
    "ladakhi": "Ladakhi",
    "bashkir": "Bashkir",
    "bosnian": "Bosnian",
    "chuvash": "Chuvash",
    "cornish": "Cornish",
    "divehi": "Divehi",
    "dyula": "Dyula",
    "friulian": "Friulian",
    "ganda": "Ganda",
    "kachin": "Kachin",
    "kamba": "Kamba",
    "kikuyu": "Kikuyu",
    "tajik": "Tajik",
    "yakut": "Yakut",
    "luganda": "Luganda",
    "tahitian": "Tahitian",
    "gitksan": "Gitksan",
    "luhya": "Luhya",
    "jamaican patois": "Jamaican Patois",
    "fulah": "Fulah",
    "rundi": "Kirundi",
    "tetum": "Tetum",
    "cherokee": "Cherokee",
    "wixarika": "Wixarika",
    "griko": "Griko",

    # Straightforward normalization fixes
    "thai": "Thai",
    "igbo": "Igbo",
    "bodo": "Bodo",
    "tulu": "Tulu",
    "mizo": "Mizo",
    "akan": "Akan",
    "yorùbá": "Yoruba",
    "dari": "Dari",
    "lao": "Lao",
    "ewe": "Ewe",
    "twi": "Twi",
    "m ̄aori": "Māori",
    "afan-oromo": "Oromo",
    "afan oromo": "Oromo",
    "afan oromo": "Oromo",
    "myanmar": "Burmese",
    "north macedonian": "Macedonian",
    "tswana": "Tswana",
    "sepedi": "Northern Sotho",
    "tshivenda": "Venda",
    "xitsonga": "Tsonga",
    "siswati": "Swati",
    "isindebele": "North Ndebele",
    "sesotho sa leboa": "Northern Sotho",
    "swati": "Swati",
    "n. sotho": "Northern Sotho",
    "s. sotho": "Southern Sotho",
    "ndebele": "Ndebele",
    "naijá": "Nigerian Pidgin",
})

ISO_CODE_MAP: Dict[str, str] = {
    "en": "English",
    "en-us": "English",
    "en-in": "English",
    "hi-in": "Hindi",
    "bn-in": "Bengali",
    "mni-in": "Manipuri",
    "deu": "German",
    "fra": "French",
    "hin": "Hindi",
    "ara": "Arabic",
    "por": "Portuguese",
    "af": "Afrikaans",
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "te": "Telugu",
    "ur": "Urdu",
    "ro": "Romanian",
    "jv": "Javanese",
    "ms": "Malay",
    "tl": "Tagalog",
    "ts": "Tsonga",
    "eu": "Basque",
    "he": "Hebrew",
    "xho": "Xhosa",
    "afr": "Afrikaans",
    "nso": "Northern Sotho",
    "sot": "Southern Sotho",
    "tsn": "Tswana",
    "ven": "Venda",
    "tso": "Tsonga",
    "ga": "Irish",
    "mt": "Maltese",
}


DIRECT_MAP.update({
    "khasi": "Khasi",
    "sundanese": "Sundanese",
    "quechua": "Quechua",
    "mongolian": "Mongolian",
    "nigerian pidgin": "Nigerian Pidgin",
    "wolaytta": "Wolaytta",
    "madurese": "Madurese",
    "tamazight": "Tamazight",
    "acehnese": "Acehnese",
    "hiligaynon": "Hiligaynon",
    "awadhi": "Awadhi",
    "gwich'in": "Gwich'In",
    "bemba": "Bemba",
    "pomak": "Pomak",
    "sylheti": "Sylheti",
    "afar": "Afar",
    "sicilian": "Sicilian",
    "angika": "Angika",
    "kabyle": "Kabyle",
    "yupik": "Yupik",
    "masai": "Masai",
    "nyishi": "Nyishi",
    "torwali": "Torwali",
    "bartangi": "Bartangi",
    "yukaghir": "Yukaghir",
    "tundra nenets": "Tundra Nenets",
    "anaang": "Anaang",
    "efik": "Efik",
    "ibibio": "Ibibio",
    "sorbian": "Sorbian",
    "tigre": "Tigre",
    "buryat": "Buryat",
    "nuosu": "Nuosu",
    "dinka": "Dinka",
    "comanche": "Comanche",
    "chittagonian": "Chittagonian",
    "tsotsitaal": "Tsotsitaal",
    "kichwa": "Kichwa",
    "dagur": "Dagur",
    "mambai": "Mambai",
    "takri": "Takri",
    "arhuaco": "Arhuaco",
    "chokwe": "Chokwe",
    "kabuverdianu": "Kabuverdianu",
    "ainu": "Ainu",
    "runyankore": "Runyankore",
    "tamang": "Tamang",
    "drenjongke": "Drenjongke",
})

DIRECT_MAP.update({
    "santhali": "Santali",
    "ho": "Ho",
    "mni": "Manipuri",
    "eesti": "Estonian",
    "malti": "Maltese",
    "newari": "Newar",
    "gaelic": "Scottish Gaelic",
    "malaysian": "Malay",
    "moldavian": "Romanian",
    "komi": "Komi-Zyrian",
    "ashaninka": "Asháninka",
    "apurina": "Apurinã",
    "bororo": "Boróro",
    "guajajara": "Guajajára",
    "kaapor": "Ka'Apor",
    "garwali": "Garhwali",
    "hariyanvi": "Haryanvi",
    "malwi": "Malvi",
    "tsonga": "Tsonga",
    "venda": "Venda",
    "north ndebele": "North Ndebele",
    "south ndebele": "South Ndebele",
    "southern ndebele": "South Ndebele",
})


PROGRAMMING_LANGUAGES = {
    "python", "c", "c#", "c sharp", "ruby", "fortran", "pascal", "perl",
    "shell", "tcl", "go", "golang", "rust", "typescript", "haskell",
    "ocaml", "erlang", "r", "scala", "julia", "solidity", "lisp"
}

SCRIPTS_AND_WRITING_SYSTEMS = {
    "cyrillic", "katakana", "tifinagh", "glagolitic", "phoenician",
    "linear b", "elder futhark", "ogham"
}

SCRIPTS_AND_WRITING_SYSTEMS.update({
    "georgian asomtavruli",
    "georgian mkhedruli",
    "kharoshthi",
    "palmyrene",
    "meroitic hieroglyphs",
    "meroitic cursive",
    "jawi",
    "lontara",
    "pegon",
    "devanagari symbol",
})

LANGUAGE_FAMILIES = {
    "slavic", "romance", "indo-european", "dravidian", "sino-tibetan",
    "austronesian languages", "uralic languages", "quechuan", "berber"
}
LANGUAGE_FAMILIES.update({
    "proto-indo-european",
    "serbo-croatian",
})

AGGREGATE_LABELS = {
    "multilingual", "multi-lingual", "code-switched", "53 languages",
    "280 different low-resource languages"
}
AGGREGATE_LABELS.update({
    "north-eastern",
    "male",
    "n",
    "uu",
})

HISTORICAL_LANGUAGES = {
    "old french", "middle french", "ancient greek", "classical arabic",
    "akkadian", "sumerian", "hittite", "hurrian", "luwian", "palaic",
    "old church slavonic", "old east slavic", "middle russian",
    "classical chinese", "ancient chinese", "old novgorodian",
}
HISTORICAL_LANGUAGES.update({
    "ottoman turkish",
    "old occitan",
    "middle low german",
    "middle high german",
    "old javanese",
    "classical armenian",
    "proto-indo-european",
    "mycenaean greek",
    "old babylonian",
    "middle babylonian peripheral",
    "standard babylonian",
    "neo-babylonian",
    "late babylonian",
    "neo-assyrian",
    "avestan",
    "carian",
    "caucasian albanian",
    "chorasmian",
    "cypriot",
    "cypro-minoan",
    "elymaic",
    "gothic",
    "hatran aramaic",
    "imperial aramaic",
    "lycian",
    "lydian",
    "mandaic",
    "manichaean",
    "nabataean",
    "old hungarian",
    "old north arabian",
    "old permiс",
    "old persian",
    "old sogdian",
    "old south arabian",
    "old turkic orkhon",
    "old turkic yenisei",
    "pahlavi",
    "pahlavi inscriptional",
    "pahlavi psalter",
    "parthian inscriptional",
    "sogdian",
    "syriac",
    "meroitic",
    "late-egyptian",
    "egyptian",
    "etruscan",
    "umbrian",
    "ugaritic",
    "old guarani",
    "classical chinese",
    "ancient chinese",
})

HISTORICAL_LANGUAGES.add("old permic")
HISTORICAL_LANGUAGES.discard("old permiс")  # only if that corrupted form appears

VARIANT_PARENT_MAP: Dict[str, Tuple[str, str]] = {
    "brazilian portuguese": ("Portuguese", "Brazilian"),
    "portuguese (brazil)": ("Portuguese", "Brazil"),
    "portuguese (brazilian)": ("Portuguese", "Brazilian"),
    "mozambique portuguese": ("Portuguese", "Mozambique"),
    "mozambican portuguese": ("Portuguese", "Mozambique"),
    "portuguese (mozambique)": ("Portuguese", "Mozambique"),

    "algerian arabic": ("Arabic", "Algerian"),
    "arabic (algerian)": ("Arabic", "Algerian"),
    "moroccan arabic": ("Arabic", "Moroccan"),
    "arabic (moroccan)": ("Arabic", "Moroccan"),
    "darija": ("Arabic", "Moroccan/Generic Darija"),
    "tunisian arabic": ("Arabic", "Tunisian"),
    "egyptian arabic": ("Arabic", "Egyptian"),
    "modern standard arabic": ("Arabic", "Modern Standard"),
    "modern standard arabic (msa)": ("Arabic", "Modern Standard"),

    "swiss german": ("German", "Swiss"),
    "austrian german": ("German", "Austrian"),
    "south tyrolean german": ("German", "South Tyrolean"),
    "bavarian": ("German", "Bavarian"),
    "alsatian": ("German", "Alsatian"),
    "alemannic": ("German", "Alemannic"),
    "low german": ("German", "Low German"),
    "low saxon": ("German", "Low Saxon"),

    "shanghainese": ("Chinese", "Shanghainese"),
    "wenzhou dialect": ("Chinese", "Wenzhou"),
    "wu chinese": ("Chinese", "Wu"),
    "hakka": ("Chinese", "Hakka"),
    "cantonese": ("Chinese", "Cantonese"),
    "taiwanese hokkien": ("Chinese", "Taiwanese Hokkien"),
    "southern min": ("Chinese", "Southern Min"),

    "roman urdu": ("Urdu", "Romanized"),
    "roman-urdu": ("Urdu", "Romanized"),
    "romanized hindi": ("Hindi", "Romanized"),
    "transliterated hindi": ("Hindi", "Romanized"),

    "african american english": ("English", "African American English"),
    "african american vernacular english": ("English", "AAVE"),
}
VARIANT_PARENT_MAP.update({
    "norwegian bokmål": ("Norwegian", "Bokmål"),
    "norwegian nynorsk": ("Norwegian", "Nynorsk"),
    "sorani kurdish": ("Kurdish", "Sorani"),
    "kurdish (sorani)": ("Kurdish", "Sorani"),
    "kurmanji kurdish": ("Kurdish", "Kurmanji"),
    "kurmanji": ("Kurdish", "Kurmanji"),
    "german (switzerland)": ("German", "Swiss"),
    "de zahrar sproche": ("German", "Regional/Dialect"),
    "standard chinese (zh)": ("Chinese", "Standard"),
    "korean (ko)": ("Korean", "Standard"),
    "tibetan (bo)": ("Tibetan", "Standard"),
    "mongolian (mn)": ("Mongolian", "Standard"),
    "uyghur (ug)": ("Uyghur", "Standard"),
    "kazakh (kk)": ("Kazakh", "Standard"),
    "zhuang (za)": ("Zhuang", "Standard"),
    "modern korean": ("Korean", "Modern"),
    "hokkien": ("Chinese", "Hokkien"),
    "simple russian": ("Russian", "Simplified Register"),
    "easy finnish": ("Finnish", "Easy/Simplified Register"),
    "indonesian colloquial": ("Indonesian", "Colloquial"),
    "acehnese (arabic script)": ("Acehnese", "Arabic Script"),
    "acehnese (latin script)": ("Acehnese", "Latin Script"),
    "banjar (arabic script)": ("Banjar", "Arabic Script"),
    "banjar (latin script)": ("Banjar", "Latin Script"),
    "kashmiri (arabic script)": ("Kashmiri", "Arabic Script"),
    "kashmiri (devanagari script)": ("Kashmiri", "Devanagari Script"),
    "eastern panjabi": ("Punjabi", "Eastern"),
    "eastern yiddish": ("Yiddish", "Eastern"),
    "northern azerbaijani": ("Azerbaijani", "Northern"),
    "northern kurdish": ("Kurdish", "Northern/Kurmanji"),
    "central aymara": ("Aymara", "Central"),
    "ayacucho quechua": ("Quechua", "Ayacucho"),
})

SIGN_LANGUAGE_MAP: Dict[str, str] = {
    "american sign language": "American Sign Language",
    "british sign language": "British Sign Language",
    "indian sign language": "Indian Sign Language",
    "german sign language": "German Sign Language",
    "japanese sign language": "Japanese Sign Language",
    "italian sign language": "Italian Sign Language",
    "korean sign language": "Korean Sign Language",
    "portuguese sign language": "Portuguese Sign Language",
    "spanish sign language": "Spanish Sign Language",
    "russian sign language": "Russian Sign Language",
    "swedish sign language": "Swedish Sign Language",
    "finnish sign language": "Finnish Sign Language",
    "moroccan sign language": "Moroccan Sign Language",
    "catalan sign language": "Catalan Sign Language",
    "peruvian sign language": "Peruvian Sign Language",
    "czech sign language": "Czech Sign Language",
    "israeli sign language": "Israeli Sign Language",
    "ghanaian sign language": "Ghanaian Sign Language",
    "french sign language": "French Sign Language",
    "mexican sign language": "Mexican Sign Language",
    "sign language": "Unspecified Sign Language",
}

MACRO_OR_REGIONAL_LABELS = {
    "gulf",
    "levantine",
    "north-african",
    "algerian dialect",
    "palestinian dialect",
}

ARABIC_REGIONAL_LABELS = {
    "gulf": ("Arabic", "Gulf"),
    "levantine": ("Arabic", "Levantine"),
    "north-african": ("Arabic", "North African"),
    "palestinian dialect": ("Arabic", "Palestinian"),
    "algerian dialect": ("Arabic", "Algerian"),
    "algerian arabic dialect": ("Arabic", "Algerian"),
    "omani arabic": ("Arabic", "Omani"),
    "tunisian-arabic": ("Arabic", "Tunisian"),
    "arabic (egyptian)": ("Arabic", "Egyptian"),
    "juba arabic": ("Arabic", "Juba Arabic"),
}

DIRECT_MAP.update({
    "latin": "Latin",
    "sanskrit": "Sanskrit",
})

HISTORICAL_LANGUAGES.update({
    "coptic",
    "ge'ez",
})

AGGREGATE_LABELS.update({
    "pidgin",
})

HISTORICAL_LANGUAGES.update({
    "old italic",
    "samaritan",
    "古汉语",
})

SCRIPTS_AND_WRITING_SYSTEMS.update({
    "hanja",
})

# ============================================================
# PARSING
# ============================================================

def parse_language_cell(value: Any) -> List[str]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [normalize_basic_text(v) for v in value if str(v).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [normalize_basic_text(v) for v in parsed if str(v).strip()]
        except Exception:
            pass

    parts = re.split(r"[;,|]", text)
    return [normalize_basic_text(p) for p in parts if normalize_basic_text(p)]


def parse_single_language_cell(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    text = normalize_basic_text(str(value))
    return [text] if text else []


# ============================================================
# CLASSIFICATION
# ============================================================

def looks_like_locale_tag(key: str) -> bool:
    return bool(re.fullmatch(r"[a-z]{2,3}-[a-z]{2,3}", key))


def is_code_mixed(key: str) -> bool:
    if "codemixed" in key or "code mixed" in key or "code-mixed" in key:
        return True
    return key in {"taglish", "hinglish", "tenglish", "arabizi", "hindi-english", "malayalam-english", "kannada-English".lower(), "tamil-english"}


def classify_and_normalize_token(raw_token: str) -> Dict[str, Any]:
    token = normalize_basic_text(raw_token)
    key = normalize_match_key(token)

    rec = {
        "raw_language_token": token,
        "normalized_language": None,
        "normalized_language_type": None,
        "parent_language": None,
        "variant_detail": None,
        "script_detail": None,
        "exclude_from_language_stats": False,
        "exclude_from_country_inference": False,
    }

    if not key:
        rec["normalized_language_type"] = "empty"
        rec["exclude_from_language_stats"] = True
        rec["exclude_from_country_inference"] = True
        return rec

    if key in SIGN_LANGUAGE_MAP or "sign language" in key:
        rec["normalized_language"] = SIGN_LANGUAGE_MAP.get(key, token)
        rec["normalized_language_type"] = "sign_language"
        rec["exclude_from_country_inference"] = True
        return rec

    if key in PROGRAMMING_LANGUAGES:
        rec["normalized_language_type"] = "programming_language"
        rec["exclude_from_language_stats"] = True
        rec["exclude_from_country_inference"] = True
        return rec

    if key in SCRIPTS_AND_WRITING_SYSTEMS:
        rec["normalized_language_type"] = "script_or_writing_system"
        rec["script_detail"] = token
        rec["exclude_from_language_stats"] = True
        rec["exclude_from_country_inference"] = True
        return rec

    if key in LANGUAGE_FAMILIES:
        rec["normalized_language_type"] = "language_family"
        rec["exclude_from_language_stats"] = True
        rec["exclude_from_country_inference"] = True
        return rec

    if key in AGGREGATE_LABELS or re.fullmatch(r"\d+\s+languages", key):
        rec["normalized_language_type"] = "aggregate_label"
        rec["exclude_from_language_stats"] = True
        rec["exclude_from_country_inference"] = True
        return rec

    if key in HISTORICAL_LANGUAGES:
        rec["normalized_language"] = token
        rec["normalized_language_type"] = "historical_language"
        rec["exclude_from_country_inference"] = True
        return rec

    if key in VARIANT_PARENT_MAP:
        parent, detail = VARIANT_PARENT_MAP[key]
        rec["normalized_language"] = parent
        rec["normalized_language_type"] = "language_variant"
        rec["parent_language"] = parent
        rec["variant_detail"] = detail
        rec["exclude_from_country_inference"] = True
        return rec
    
    if key in ARABIC_REGIONAL_LABELS:
        parent, detail = ARABIC_REGIONAL_LABELS[key]
        rec["normalized_language"] = parent
        rec["normalized_language_type"] = "language_variant"
        rec["parent_language"] = parent
        rec["variant_detail"] = detail
        rec["exclude_from_country_inference"] = True
        return rec

    if key in ISO_CODE_MAP:
        mapped = ISO_CODE_MAP[key]
        rec["normalized_language"] = mapped
        rec["normalized_language_type"] = "iso_code"
        rec["parent_language"] = mapped
        return rec

    if looks_like_locale_tag(key):
        base = key.split("-")[0]
        if base in ISO_CODE_MAP:
            rec["normalized_language"] = ISO_CODE_MAP[base]
            rec["normalized_language_type"] = "locale_tag"
            rec["parent_language"] = ISO_CODE_MAP[base]
            rec["exclude_from_country_inference"] = True
            return rec

    if key in DIRECT_MAP:
        mapped = DIRECT_MAP[key]
        rec["normalized_language"] = mapped
        rec["normalized_language_type"] = "canonical_language"
        rec["parent_language"] = mapped
        return rec

    if is_code_mixed(key):
        rec["normalized_language_type"] = "code_mixed_or_transliterated"
        rec["exclude_from_country_inference"] = True
        return rec

    if "simplified chinese" in key or "traditional chinese" in key:
        rec["normalized_language"] = "Chinese"
        rec["normalized_language_type"] = "script_or_written_variant"
        rec["parent_language"] = "Chinese"
        rec["exclude_from_country_inference"] = True
        return rec

    rec["normalized_language"] = token
    rec["normalized_language_type"] = "canonical_language"
    rec["parent_language"] = token
    rec["exclude_from_country_inference"] = True
    return rec

# ============================================================
# ROW PROCESSING
# ============================================================

def process_language_tokens(tokens: List[str]) -> Dict[str, Any]:
    records = [classify_and_normalize_token(tok) for tok in tokens]

    canonical = []
    variants = []
    sign_languages = []
    excluded = []

    for rec in records:
        rec_type = rec["normalized_language_type"]
        norm = rec["normalized_language"]

        if rec_type in {"canonical_language", "iso_code"} and norm:
            canonical.append(norm)
        elif rec_type == "language_variant" and rec["parent_language"]:
            canonical.append(rec["parent_language"])
            variants.append(rec["raw_language_token"])
        elif rec_type == "sign_language" and norm:
            sign_languages.append(norm)
        elif rec_type != "empty":
            excluded.append(rec["raw_language_token"])

    canonical = dedupe_preserve_order(canonical)
    variants = dedupe_preserve_order(variants)
    sign_languages = dedupe_preserve_order(sign_languages)
    excluded = dedupe_preserve_order(excluded)

    return {
        "canonical_language_list": canonical,
        "variant_language_list": variants,
        "sign_language_list": sign_languages,
        "excluded_language_like_tokens": excluded,
        "language_type_records": records,
    }


# ============================================================
# FILE PROCESSING
# ============================================================

def process_file(file_cfg: Dict[str, str]) -> None:
    name = file_cfg["name"]
    input_csv = Path(file_cfg["input_csv"])
    language_column = file_cfg["language_column"]
    mode = file_cfg["mode"]

    if not input_csv.exists():
        print(f"[SKIP] {name}: file not found -> {input_csv}")
        return

    df = pd.read_csv(input_csv)

    if language_column not in df.columns:
        print(f"[SKIP] {name}: missing column '{language_column}'")
        print(f"Available columns: {list(df.columns)}")
        return

    if mode == "list":
        parsed = df[language_column].apply(parse_language_cell)
    elif mode == "single":
        parsed = df[language_column].apply(parse_single_language_cell)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    processed = parsed.apply(process_language_tokens)

    df["parsed_language_tokens"] = parsed.apply(lambda x: json.dumps(x, ensure_ascii=False))
    df["canonical_language_list_cleaned"] = processed.apply(lambda x: json.dumps(x["canonical_language_list"], ensure_ascii=False))
    df["variant_language_list_cleaned"] = processed.apply(lambda x: json.dumps(x["variant_language_list"], ensure_ascii=False))
    df["sign_language_list_cleaned"] = processed.apply(lambda x: json.dumps(x["sign_language_list"], ensure_ascii=False))
    df["excluded_language_like_tokens_cleaned"] = processed.apply(lambda x: json.dumps(x["excluded_language_like_tokens"], ensure_ascii=False))
    df["language_type_records_cleaned"] = processed.apply(lambda x: json.dumps(x["language_type_records"], ensure_ascii=False))

    df["canonical_language_count_cleaned"] = processed.apply(lambda x: len(x["canonical_language_list"]))
    df["variant_language_count_cleaned"] = processed.apply(lambda x: len(x["variant_language_list"]))
    df["sign_language_count_cleaned"] = processed.apply(lambda x: len(x["sign_language_list"]))
    df["excluded_language_like_token_count_cleaned"] = processed.apply(lambda x: len(x["excluded_language_like_tokens"]))
    df["canonical_language_string_cleaned"] = processed.apply(lambda x: "; ".join(x["canonical_language_list"]))

    # Helpful for exploded tables
    if mode == "single":
        df["language_cleaned"] = processed.apply(
            lambda x: x["canonical_language_list"][0] if x["canonical_language_list"] else None
        )
        df["language_cleaned_type"] = processed.apply(
            lambda x: x["language_type_records"][0]["normalized_language_type"] if x["language_type_records"] else None
        )
        df["language_excluded_from_country_inference"] = processed.apply(
            lambda x: x["language_type_records"][0]["exclude_from_country_inference"] if x["language_type_records"] else True
        )

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = out_dir / f"{name}_second_stage_cleaned.csv"
    df.to_csv(cleaned_path, index=False)

    # audit tables
    all_records = []
    for recs in processed.apply(lambda x: x["language_type_records"]):
        all_records.extend(recs)

    audit_dir = out_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if all_records:
        audit_df = pd.DataFrame(all_records)
        audit_df.to_csv(audit_dir / f"{name}_token_audit.csv", index=False)

        type_counts = audit_df["normalized_language_type"].value_counts(dropna=False).reset_index()
        type_counts.columns = ["normalized_language_type", "count"]
        type_counts.to_csv(audit_dir / f"{name}_type_counts.csv", index=False)

        unresolved = (
            audit_df[audit_df["normalized_language_type"] == "canonical_language_needs_review"]["raw_language_token"]
            .value_counts()
            .reset_index()
        )
        unresolved.columns = ["raw_language_token", "count"]
        unresolved.to_csv(audit_dir / f"{name}_needs_review_counts.csv", index=False)

    print(f"[DONE] {name} -> {cleaned_path}")


def main() -> None:
    for cfg in FILE_CONFIGS:
        process_file(cfg)


if __name__ == "__main__":
    main()
from __future__ import annotations

"""Informasi negara ASEAN untuk demo ASR → TTS.

Tujuan file ini:
- Menyimpan dictionary `COUNTRY_INFO` (lokasi/kawasan, ibu kota, mata uang).
- Menyediakan helper untuk format teks (GUI) dan format kalimat (TTS).

Catatan presentasi:
- Data dibuat ringkas dan mudah dibacakan oleh pyttsx3.
- Kunci dictionary mengikuti label model ASR (contoh: "Brunei", "Filipina").
"""

from typing import TypedDict


class CountryInfo(TypedDict):
    lokasi: str
    ibu_kota: str
    mata_uang: str


# === Requirement UAS: COUNTRY_INFO untuk 10 negara ASEAN ===
COUNTRY_INFO: dict[str, CountryInfo] = {
    "Indonesia": {
        "lokasi": "Asia Tenggara",
        "ibu_kota": "Jakarta",
        "mata_uang": "Rupiah",
    },
    "Malaysia": {
        "lokasi": "Asia Tenggara",
        "ibu_kota": "Kuala Lumpur",
        "mata_uang": "Ringgit Malaysia",
    },
    "Singapura": {
        "lokasi": "Asia Tenggara",
        "ibu_kota": "Singapura",
        "mata_uang": "Dolar Singapura",
    },
    "Thailand": {
        "lokasi": "Asia Tenggara",
        "ibu_kota": "Bangkok",
        "mata_uang": "Baht",
    },
    "Vietnam": {
        "lokasi": "Asia Tenggara",
        "ibu_kota": "Hanoi",
        "mata_uang": "Dong Vietnam",
    },
    "Laos": {
        "lokasi": "Asia Tenggara",
        "ibu_kota": "Vientiane",
        "mata_uang": "Kip Laos",
    },
    "Myanmar": {
        "lokasi": "Asia Tenggara",
        "ibu_kota": "Naypyidaw",
        "mata_uang": "Kyat Myanmar",
    },
    "Filipina": {
        "lokasi": "Asia Tenggara",
        "ibu_kota": "Manila",
        "mata_uang": "Peso Filipina",
    },
    "Brunei": {
        "lokasi": "Asia Tenggara",
        "ibu_kota": "Bandar Seri Begawan",
        "mata_uang": "Dolar Brunei",
    },
    "Kamboja": {
        "lokasi": "Asia Tenggara",
        "ibu_kota": "Phnom Penh",
        "mata_uang": "Riel Kamboja",
    },
}


def _normalize_label(label: str) -> str:
    """Normalisasi label agar matching dengan key dictionary.

    Model ASR biasanya mengeluarkan label yang sudah konsisten.
    Tapi normalisasi ini membantu kalau ada spasi/format kecil.
    """

    return (label or "").strip()


def get_country_info(label: str) -> CountryInfo | None:
    """Ambil info negara dari label prediksi ASR."""

    key = _normalize_label(label)
    return COUNTRY_INFO.get(key)


def format_country_info_text(label: str) -> str:
    """Format untuk ditampilkan di GUI (multi-line, gaya narasi).

    Format ini sengaja dibuat seperti naskah presentasi UAS:
    - baris 1: "<Negara> adalah negara di <Lokasi>."
    - baris 2: "Ibu kotanya <Ibu kota>."
    - baris 3: "Mata uangnya <Mata uang>."
    """

    info = get_country_info(label)
    if not info:
        return "-"

    key = _normalize_label(label)
    return (
        f"{key} adalah negara di {info['lokasi']}.\n"
        f"Ibu kotanya {info['ibu_kota']}.\n"
        f"Mata uangnya {info['mata_uang']}."
    )


def format_country_info_tts_text(label: str) -> str:
    """Format teks yang dikirim ke textbox TTS.

    Dibuat sama dengan GUI agar konsisten:
    - Saat ASR sukses memprediksi negara, textbox TTS otomatis berisi info ini.
    """

    return format_country_info_text(label)


def format_country_info_speech(label: str) -> str:
    """Format kalimat untuk dibacakan TTS.

    Contoh:
    "Brunei adalah negara di Asia Tenggara. Ibu kotanya Bandar Seri Begawan dan mata uangnya Dolar Brunei."
    """

    info = get_country_info(label)
    if not info:
        # Fallback: jika label tidak ada di dictionary, minimal sebutkan labelnya.
        return _normalize_label(label)

    key = _normalize_label(label)
    return (
        f"{key} adalah negara di {info['lokasi']}. "
        f"Ibu kotanya {info['ibu_kota']} dan mata uangnya {info['mata_uang']}."
    )

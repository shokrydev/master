# Geographic utilities for GAIA dataset analysis
# Maps (lat, lon) coordinates to continents using offline reverse geocoding

from collections.abc import Sequence

import reverse_geocoder as rg

# Continent → ISO 3166-1 alpha-2 codes (UN M49 standard)
# Inverted into _CC_TO_CONTINENT at module load
_CONTINENT_COUNTRIES = {
    "Africa": (
        "AO BF BI BJ BW CD CF CG CI CM CV DJ DZ EG EH ER ET GA GH GM GN GQ "
        "GW KE KM LR LS LY MA MG ML MR MU MW MZ NA NE NG RE RW SC SD SH SL "
        "SN SO SS ST SZ TD TG TN TZ UG YT ZA ZM ZW"
    ),
    "Asia": (
        "AE AF AM AZ BD BH BN BT CN CY GE HK ID IL IN IQ IR JO JP KG KH KP "
        "KR KW KZ LA LB LK MM MN MO MV MY NP OM PH PK PS QA SA SG SY TH TJ "
        "TL TM TR TW UZ VN YE"
    ),
    "Europe": (
        "AD AL AT AX BA BE BG BY CH CZ DE DK EE ES FI FO FR GB GG GI GR HR "
        "HU IE IM IS IT JE LI LT LU LV MC MD ME MK MT NL NO PL PT RO RS RU "
        "SE SI SK SM UA VA XK"
    ),
    "North America": (
        "AG AI AW BB BL BM BQ BS BZ CA CR CU CW DM DO GD GL GP GT HN HT JM "
        "KN KY LC MF MQ MS MX NI PA PM PR SV SX TC TT US VC VG VI"
    ),
    "South America": "AR BO BR CL CO EC FK GF GY PE PY SR UY VE",
    "Oceania": (
        "AS AU CK FJ FM GU KI MH MP NC NF NR NU NZ PF PG PN PW SB TK TO TV "
        "UM VU WF WS"
    ),
    "Antarctica": "AQ BV GS HM TF",
}

_CC_TO_CONTINENT = {}
for _continent, _codes in _CONTINENT_COUNTRIES.items():
    for _cc in _codes.split():
        _CC_TO_CONTINENT[_cc] = _continent


def assign_continent(lat: float, lon: float) -> str | None:
    """Assign continent from a single (lat, lon) coordinate."""
    result = rg.search([(lat, lon)])[0]
    return _CC_TO_CONTINENT.get(result["cc"])


def assign_continents(
    lats: Sequence[float], lons: Sequence[float]
) -> list[str | None]:
    """Assign continents for a batch of coordinates (single k-d tree lookup)."""
    coords = list(zip(lats, lons, strict=True))
    results = rg.search(coords)
    return [_CC_TO_CONTINENT.get(r["cc"]) for r in results]

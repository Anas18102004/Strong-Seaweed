# -*- coding: utf-8 -*-
import json
import re
import serve_kappaphycus_api as s

points = [
    ("Agatti Island", "10°51'21.60\"N", "72°11'34.80\"E"),
    ("Amini Island", "11°7'19.79\"N", "72°43'30.41\"E"),
    ("Andrott Island", "10°48'41.51\"N", "73°40'38.53\"E"),
    ("Bitra Island", "11°35'55.11\"N", "72°11'11.88\"E"),
    ("Balliyapani reef", "12°23'52.80\"N", "71°53'54.96\"E"),
    ("Bangaram Island", "10°56'17.95\"N", "72°17'15.92\"E"),
    ("Kodi point rock Islet", "8°19'26.90\"N", "73°4'39.33\"E"),
    ("Koditala Islet", "10°6'34.63\"N", "73°39'10.72\"E"),
    ("Viringili Islet", "8°16'41.89\"N", "73°0'41.38\"E"),
    ("Katchall Island", "07°56'29.46\"N", "93°24'02.23\"E"),
    ("Koswari Tivu", "08°52'12.00\"N", "78°13'31.08\"E"),
    ("Krusadai Island", "09°14'49.20\"N", "79°12'41.04\"E"),
    ("Anaipar Tivu", "09°09'10.80\"N", "78°41'41.64\"E"),
    ("Appa Tivu A", "09°09'44.05\"N", "78°49'14.01\"E"),
    ("Appa Tivu B", "09°10'03.54\"N", "78°49'37.39\"E"),
    ("Kodiyampalayam", "11°23'00.42\"N", "79°48'59.32\"E"),
    ("Ajad Tapu", "22°22'49.08\"N", "69°19'54.84\"E"),
    ("Alia Bet", "21°35'20.04\"N", "72°39'24.84\"E"),
    ("Amudi Beli", "22°32'29.04\"N", "69°57'00.36\"E"),
    ("Asab Island", "22°23'46.68\"N", "69°12'20.52\"E"),
]


def dms_to_dd(value: str) -> float:
    m = re.match(r"\s*(\d+)[°](\d+)[']([0-9.]+)[\"]([NSEW])\s*", value)
    if not m:
        raise ValueError(f"Bad DMS: {value}")
    d, mi, se, hemi = int(m.group(1)), int(m.group(2)), float(m.group(3)), m.group(4)
    out = d + mi / 60 + se / 3600
    return -out if hemi in {"S", "W"} else out


rows = []
for site, lat_dms, lon_dms in points:
    lat = dms_to_dd(lat_dms)
    lon = dms_to_dd(lon_dms)
    pred = s.predict_point(lat, lon)
    rows.append(
        {
            "site": site,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "probability_percent": pred["kappaphycus"]["probability_percent"],
            "label": pred["kappaphycus"]["pred_label"],
            "priority": pred["kappaphycus"]["priority"],
            "nearest_grid_km": round(pred["nearest_grid"]["distance_km"], 2),
        }
    )

print(json.dumps(rows, indent=2))

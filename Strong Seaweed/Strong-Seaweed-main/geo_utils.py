import numpy as np

try:
    from pyproj import Transformer as _Transformer
except Exception:
    _Transformer = None


class _ApproxTransformer:
    """Fallback lon/lat -> pseudo-meters transform when pyproj is unavailable."""

    def transform(self, lon, lat):
        lon_arr = np.asarray(lon, dtype=np.float64)
        lat_arr = np.asarray(lat, dtype=np.float64)
        lat0 = float(np.nanmean(lat_arr)) if lat_arr.size else 0.0
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(lat0)))
        x = lon_arr * m_per_deg_lon
        y = lat_arr * m_per_deg_lat
        return x, y


def make_metric_transformer():
    if _Transformer is not None:
        return _Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    return _ApproxTransformer()

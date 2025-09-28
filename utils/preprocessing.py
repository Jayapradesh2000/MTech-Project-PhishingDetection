import re
from urllib.parse import urlparse

IP_PATTERN = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$"  # simple IPv4 in host
)

def has_ip(host: str) -> bool:
    if not host:
        return False
    # remove port if present
    host_only = host.split(':')[0]
    return bool(IP_PATTERN.match(host_only))

def extract_features_from_url(url: str) -> dict:
    """
    Basic features extracted from a URL string.
    Returns a dict with numeric features.
    """
    if not isinstance(url, str):
        url = str(url)

    parsed = urlparse(url)
    host = parsed.netloc.lower()

    features = {
        "URL_Length": len(url),
        "Has_IP": 1 if has_ip(host) else 0,
        "Prefix_Suffix": 1 if '-' in host else 0,   # hyphen in domain
        "Count_Dots": host.count('.'),
        "Has_At": 1 if '@' in url else 0,
        "Count_Hyphens": url.count('-'),
        "Has_HTTPS": 1 if parsed.scheme == "https" else 0,
        "Domain_Tokens": len([t for t in host.split('.') if t]),
    }
    return features

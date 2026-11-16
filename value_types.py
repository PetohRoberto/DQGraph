"""
value_types.py — What a property value really is, whatever its storage type.

CSV imports often store numbers as strings ("1995", "7.5") and dates as strings
("1995-11-22") or Neo4j temporal values. The schema explorer, the ML features
and the accuracy degrader all use these helpers so that such properties are
treated as numbers everywhere.
"""
import datetime
import math
import re

# A property is numeric/temporal when at least this share of its sampled
# non-null values converts.
CONVERTIBLE_SHARE = 0.8
# Integer properties with at most this many distinct values can also be
# classification targets (ratings, years of a small range, levels...).
DISCRETE_MAX_DISTINCT = 20

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?")


def _decimal_year(d: datetime.date) -> float:
    start = datetime.date(d.year, 1, 1)
    days = (datetime.date(d.year + 1, 1, 1) - start).days
    return d.year + (d.toordinal() - start.toordinal()) / days


def _parse_number(s: str):
    s = s.strip()
    # Codes with leading zeros ("0114709", zip codes) are identifiers, not numbers.
    if len(s) > 1 and s[0] == "0" and s[1].isdigit():
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def _as_date(v):
    if hasattr(v, "to_native"):           # neo4j.time.Date / DateTime
        try:
            v = v.to_native()
        except Exception:
            return None
    if isinstance(v, datetime.datetime):
        return v.date()
    if isinstance(v, datetime.date):
        return v
    if isinstance(v, str) and _ISO_DATE.match(v.strip()):
        try:
            return datetime.date.fromisoformat(v.strip()[:10])
        except ValueError:
            return None
    return None


def value_kind(v) -> str:
    """'boolean' | 'numeric' | 'temporal' | 'string' | 'list' | 'other'."""
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "numeric" if math.isfinite(v) else "other"
    if isinstance(v, (list, tuple)):
        return "list"
    if _as_date(v) is not None:
        return "temporal"
    if isinstance(v, str):
        if v.strip().lower() in ("true", "false"):
            return "boolean"
        return "numeric" if _parse_number(v) is not None else "string"
    return "other"


def to_number(v):
    """Float value of a numeric or temporal value (dates → decimal year), else None."""
    kind = value_kind(v)
    if kind == "numeric":
        return float(v) if not isinstance(v, str) else _parse_number(v)
    if kind == "temporal":
        return _decimal_year(_as_date(v))
    return None


def classify(samples: list, n_distinct: int) -> str:
    """
    Type of a property from sampled non-null values:
    'boolean', 'numeric', 'numeric_discrete', 'temporal', 'categorical',
    'string' or 'other'.
    """
    samples = [s for s in samples if s is not None]
    if not samples:
        return "other"
    kinds = [value_kind(s) for s in samples]
    n = len(samples)
    share = lambda k: kinds.count(k) / n

    if share("boolean") > 0.5:
        return "boolean"
    if share("numeric") >= CONVERTIBLE_SHARE:
        nums = [to_number(s) for s, k in zip(samples, kinds) if k == "numeric"]
        if n_distinct <= DISCRETE_MAX_DISTINCT and all(x == int(x) for x in nums):
            return "numeric_discrete"
        return "numeric"
    if share("temporal") >= CONVERTIBLE_SHARE:
        return "temporal"
    if share("string") + share("numeric") > 0.5:
        return "categorical" if n_distinct <= 50 else "string"
    return "other"


def _iso_forms_of_datetime(dt: datetime.datetime) -> list[str]:
    forms = [
        dt.date().isoformat(),                                        # 2020-01-01
        dt.replace(tzinfo=None).isoformat(timespec="seconds"),       # 2020-01-01T10:00:00
        dt.replace(tzinfo=None).isoformat(),                         # ...T10:00:00.123000
    ]
    if dt.tzinfo is not None:
        for full in (dt.isoformat(timespec="seconds"), dt.isoformat()):
            forms.append(full)                                        # ...T10:00:00+01:00
            if full.endswith("+00:00"):
                forms.append(full[:-6] + "Z")                         # ...T10:00:00Z
    return list(dict.fromkeys(forms))                                 # drop duplicates


def regex_texts(v) -> list[str]:
    """
    Text forms of a value that a conformity regex is tested against, so that
    conformity does not depend on how the database stores the value.

    - String: the string itself, unchanged. Its syntax is exactly what
      conformity checks, so "22/11/1995" or "1995-11-22@1" do not conform
      to an ISO 8601 date regex.
    - Number or boolean: its usual text ("1995", "7.5", "true").
    - Native temporal value (Neo4j Date / DateTime / LocalDateTime / Time, or
      a Python date / datetime / time): its ISO 8601 forms, rebuilt here
      because Neo4j's toString drops zero seconds ("2020-01-01T10:00Z").
      A date-time gives its date ("2020-01-01"), its date-time to the second
      ("2020-01-01T10:00:00"), and its full value with fraction and offset;
      so a date stored as a DateTime conforms to a date regex.
    - List, map or null: no form, i.e. never conforming.
    """
    if v is None or isinstance(v, (list, tuple, dict)):
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, bool):
        return ["true" if v else "false"]
    if isinstance(v, int):
        return [str(v)]
    if isinstance(v, float):
        if not math.isfinite(v):
            return [str(v)]
        return [repr(v), str(int(v))] if v.is_integer() else [repr(v)]
    if hasattr(v, "to_native"):           # neo4j.time.Date / DateTime / Time
        try:
            v = v.to_native()
        except Exception:
            return [str(v)]
    if isinstance(v, datetime.datetime):
        return _iso_forms_of_datetime(v)
    if isinstance(v, datetime.date):
        return [v.isoformat()]
    if isinstance(v, datetime.time):
        forms = [v.replace(tzinfo=None).isoformat(timespec="seconds"),
                 v.replace(tzinfo=None).isoformat()]
        if v.tzinfo is not None:
            forms.append(v.isoformat())
        return forms
    return [str(v)]                       # durations, points...


def matches_regex(v, pattern: "re.Pattern") -> bool:
    """True if one text form of v (see regex_texts) fully matches pattern,
    like Cypher's =~."""
    return any(pattern.fullmatch(t) for t in regex_texts(v))


NUMERIC_TYPES     = ("numeric", "numeric_discrete", "temporal")
CATEGORICAL_TYPES = ("categorical", "boolean", "numeric_discrete")

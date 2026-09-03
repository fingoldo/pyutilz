"""Time, date and timestamp conversion/formatting helpers.

Split out of the historical flat ``pyutilz.core.pythonlib`` module; re-exported from the
package ``__init__`` to preserve the public import surface.
"""

import re

from ._common import Optional, date, datetime, logger, time, timezone

# ----------------------------------------------------------------------------------------------------------------------------
# Time & dates
# ----------------------------------------------------------------------------------------------------------------------------


def utc_to_local(utc_dt):
    """Converts a naive UTC datetime to a timezone-aware datetime in the local system timezone."""
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)


def utc_ts_2_locstr(
    inp_dt: str,
    dst: Optional[int] = None,
    dst_names: Optional[dict] = None,
    input_date_format="%Y-%m-%dT%H:%M:%S.%f",
    output_date_format="%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Format UTC timestamp & optionally calculate time passed

    >utc_ts_2_locstr(inp_dt="2021-09-22T15:14:34.532707", dst=60)
    '2021-09-22 18:14:34 (102 мин. тому назад)'

    """
    if dst_names is None:
        dst_names = {
            1: "сек",
            60: "мин",
            60 * 60: "ч",
            60 * 60 * 24: "дн",
            60 * 60 * 24 * 30: "мес",
            60 * 60 * 24 * 365: "гд",
        }
    if inp_dt is None or inp_dt == "":
        return ""
    utc_dt = datetime.strptime(inp_dt, input_date_format)  # inp_dt has no tz component (input_date_format has no %z); result is treated as naive-UTC by utc_to_local() below
    res = f"{utc_to_local(utc_dt).strftime(output_date_format)}"

    if dst:
        if isinstance(dst, int):
            if dst > 0:
                if dst in dst_names:
                    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)  # must stay naive to subtract against utc_dt (naive-UTC from strptime above); switching to an aware value would raise offset-naive/-aware TypeError
                    res += f" ({(utc_now - utc_dt).total_seconds() // dst:.0f} {dst_names.get(dst)}. тому назад)"
    return res


# A timezone offset is a trailing +HH:MM / -HH:MM, ANCHORED at the end of the string. Searching for
# a bare "+"/"-" anywhere matched the date's own hyphens, so a timestamp with NO offset had every
# colon stripped out of its time ('2020-02-20T11:54:00' -> '2020-02-20T115400').
_TZ_OFFSET_RE = re.compile(r"([+-]\d{2}):(\d{2})$")


def read_timezoned_ts(inp):
    """Drop the colon from a trailing timezone offset; a string without one is returned unchanged.

    >>> read_timezoned_ts('2020-02-20T11:54:00.000-07:00')
    '2020-02-20T11:54:00.000-0700'
    >>> read_timezoned_ts('2020-02-20T11:54:00')
    '2020-02-20T11:54:00'
    """
    return _TZ_OFFSET_RE.sub(r"\1\2", inp)


def datetime_to_unix_ts(dt):
    """Converts a naive datetime (treated as UTC) into a whole-second Unix timestamp."""
    from datetime import datetime

    return int((dt - datetime(1970, 1, 1)).total_seconds())  # noqa: DTZ001 -- epoch reference must stay naive to match `dt`, which is documented/treated as a naive-UTC datetime


def get_utc_unix_ts_seconds() -> int:
    """Returns the current UTC Unix timestamp, in whole seconds."""
    return int(time.time())


def get_utc_unix_ts_milliseconds() -> int:
    """Returns the current UTC Unix timestamp, in whole milliseconds."""
    return int(time.time() * 1000)


def imitate_delay(
    min_delay_seconds: float,
    max_delay_seconds: float,
    last_call_ts: Optional[datetime] = None,
    b_force: Optional[bool] = True,
    big_delay_prob: Optional[float] = None,
    big_delay_multiplier: Optional[float] = 10,
) -> datetime:
    """
    Waits random time interval (delay) since the last action.
    >>> last_call_ts=None;last_call_ts=imitate_delay(2,4,last_call_ts);last_call_ts=imitate_delay(2,4,last_call_ts);

    """
    from datetime import datetime, timezone
    from random import uniform, random
    from time import sleep

    if min_delay_seconds >= max_delay_seconds:
        logger.warning("min_delay_seconds of %s >= max_delay_seconds of %s!", min_delay_seconds, max_delay_seconds)
        max_delay_seconds = min_delay_seconds * 2

    if last_call_ts or b_force:
        random_delay = uniform(min_delay_seconds, max_delay_seconds)  # nosec B311 - non-cryptographic use: picks a human-facing sleep delay to imitate pacing, not a security token
        if big_delay_prob:
            if big_delay_prob > 0:
                if random() < big_delay_prob:  # nosec B311 - non-cryptographic use: coin-flip to decide whether to lengthen the imitated delay, not security-sensitive
                    random_delay = random_delay * (big_delay_multiplier if big_delay_multiplier is not None else 10)
        if b_force and (last_call_ts is None):
            cur_delay = 0.0
        else:
            assert last_call_ts is not None  # guaranteed by the `if last_call_ts or b_force:` guard above combined with this branch
            cur_delay = (datetime.now(timezone.utc).replace(tzinfo=None) - last_call_ts).total_seconds()  # last_call_ts is a caller-supplied naive-UTC datetime (public API contract, see tests); must stay naive to subtract
        if cur_delay < random_delay:
            logger.debug("Sleeping %.2f sec.", random_delay - cur_delay)
            sleep(random_delay - cur_delay)
    return datetime.now(timezone.utc).replace(tzinfo=None)  # return value feeds back into this function's own last_call_ts (naive-UTC contract, see tests); must stay naive


def weekofmonth(date: date):
    """Returns the 1-based week-of-month number for `date` (7-day buckets starting at day 1)."""
    return (date.day - 1) // 7 + 1


def datetime_to_utc_timestamp(dt):
    """Converts a datetime object to UTC timestamp

    naive datetime will be considered UTC.

    """
    import calendar

    return calendar.timegm(dt.utctimetuple())


def age(dob: date) -> int:
    """Returns the whole number of years elapsed between `dob` and today."""
    today = date.today()  # age is a calendar-date-only computation (no time-of-day), naive date is the correct type here
    years = today.year - dob.year
    if today.month < dob.month or (today.month == dob.month and today.day < dob.day):
        years -= 1
    return years

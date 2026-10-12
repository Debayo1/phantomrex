import sys
import time
import json
import os
import requests
from datetime import datetime, timezone, timedelta

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CACHE_FILE = "economic_calendar_cache.json"
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


class NewsFilter:
    """
    Automated High-Impact Economic News Calendar Filter.
    Pauses new trade execution during volatile macroeconomic announcements (CPI, NFP, FOMC, PPI, Fed Speeches).
    """
    def __init__(self, pre_news_min=20, post_news_min=15, enabled=True):
        self.pre_news_min = pre_news_min
        self.post_news_min = post_news_min
        self.enabled = enabled
        self.events = []
        self.last_fetch_time = 0
        self.fetch_interval = 4 * 3600  # Refresh calendar every 4 hours
        self._load_cached_events()
        self.refresh_calendar()

    def _load_cached_events(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.events = data.get("events", [])
                    self.last_fetch_time = data.get("timestamp", 0)
            except Exception as e:
                print(f"⚠️ Could not load news cache: {e}")

    def _save_cached_events(self):
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({"timestamp": self.last_fetch_time, "events": self.events}, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save news cache: {e}")

    def refresh_calendar(self, force=False):
        now = time.time()
        if not force and (now - self.last_fetch_time < self.fetch_interval) and self.events:
            return

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            r = requests.get(CALENDAR_URL, headers=headers, timeout=10)
            if r.status_code == 200:
                raw_events = r.json()
                filtered = []
                for ev in raw_events:
                    # Filter for High Impact USD news
                    impact = str(ev.get("impact", "")).strip().title()
                    country = str(ev.get("country", "")).strip().upper()
                    if impact == "High" and country in ["USD", "ALL"]:
                        filtered.append({
                            "title": ev.get("title", "High-Impact USD Event"),
                            "country": country,
                            "date": ev.get("date", ""),
                            "impact": impact,
                            "forecast": ev.get("forecast", ""),
                            "previous": ev.get("previous", "")
                        })
                self.events = filtered
                self.last_fetch_time = now
                self._save_cached_events()
                print(f"📰 Economic News Calendar updated: {len(self.events)} High-Impact USD events tracked.")
        except Exception as e:
            print(f"⚠️ Failed to refresh economic calendar feed: {e}")

    def check_news_block(self):
        """
        Returns:
            (is_blocked: bool, event_summary: str, minutes_to_event: float)
        """
        if not self.enabled:
            return False, "", 0.0

        self.refresh_calendar()

        now_utc = datetime.now(timezone.utc)

        for ev in self.events:
            date_str = ev.get("date", "")
            if not date_str:
                continue

            try:
                # Parse ISO-8601 date string (e.g. 2026-10-07T08:30:00-04:00)
                ev_time = datetime.fromisoformat(date_str)
                # Convert to UTC
                ev_time_utc = ev_time.astimezone(timezone.utc)
                diff_sec = (ev_time_utc - now_utc).total_seconds()
                diff_min = diff_sec / 60.0

                # Pre-news window: within pre_news_min before event (diff_min between 0 and pre_news_min)
                # Post-news window: within post_news_min after event (diff_min between -post_news_min and 0)
                if -self.post_news_min <= diff_min <= self.pre_news_min:
                    title = ev.get("title", "High-Impact Economic Release")
                    if diff_min >= 0:
                        status = f"🔴 UPCOMING ({diff_min:.0f}m until release)"
                    else:
                        status = f"🟠 RECENT ({abs(diff_min):.0f}m elapsed, settling volatility)"

                    summary = f"📰 <b>High-Impact USD Event:</b> {title} ({status})"
                    return True, summary, diff_min
            except Exception:
                continue

        return False, "", 0.0

    def get_upcoming_events(self, hours_ahead=24):
        """Returns list of upcoming events in the next N hours."""
        self.refresh_calendar()
        now_utc = datetime.now(timezone.utc)
        upcoming = []

        for ev in self.events:
            date_str = ev.get("date", "")
            if not date_str:
                continue
            try:
                ev_time = datetime.fromisoformat(date_str).astimezone(timezone.utc)
                diff_hours = (ev_time - now_utc).total_seconds() / 3600.0
                if 0 <= diff_hours <= hours_ahead:
                    upcoming.append({
                        "title": ev.get("title"),
                        "time_utc": ev_time.strftime("%H:%M UTC (%b %d)"),
                        "hours_away": round(diff_hours, 1),
                        "forecast": ev.get("forecast", "-"),
                        "previous": ev.get("previous", "-")
                    })
            except Exception:
                continue

        return sorted(upcoming, key=lambda x: x["hours_away"])

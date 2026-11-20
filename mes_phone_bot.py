"""MES phone-controlled paper signal monitor (research only).

Runs one check and exits. GitHub Actions can schedule repeated checks.
It never connects to a broker or executes paper/live orders.
Yahoo Finance price updates can be delayed, stale, or wrong.
"""

import csv
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

SYMBOL = 'MES=F'
TZ = ZoneInfo('America/Chicago')
STATE = Path('mes_state.json')
SIGNALS = Path('mes_signals.csv')
SESSION_START = time(8, 30)
SESSION_END = time(15, 10)  # includes a short after-close buffer
MAX_STALE_MINUTES = 45


def load_state():
    if not STATE.exists():
        return {}
    return json.loads(STATE.read_text(encoding='utf-8'))


def save_state(state):
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def download_bars():
    bars = yf.download(
        SYMBOL, period='5d', interval='5m', auto_adjust=False,
        progress=False, threads=False, multi_level_index=False
    )
    if bars is None or bars.empty:
        raise RuntimeError('No MES data returned; no signals generated.')
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    bars = bars[['Open', 'High', 'Low', 'Close']].dropna().sort_index()
    if bars.index.tz is None:
        # yfinance normally delivers tz-aware bars; cannot safely guess the zone.
        raise RuntimeError('Price data has no timezone; refusing to process.')
    bars.index = bars.index.tz_convert(TZ)
    bars = bars[~bars.index.duplicated(keep='last')]
    return bars


def get_closed_session_bars(bars, now_utc):
    now_ct = now_utc.astimezone(TZ)
    # Ignore current potentially incomplete 5-minute bar, with 1 minute cushion.
    cutoff = pd.Timestamp(now_utc - timedelta(minutes=6)).tz_convert(TZ)
    bars = bars.loc[bars.index <= cutoff].copy()
    bars = bars.between_time('08:30', '14:55')
    bars = bars[bars.index.dayofweek < 5]
    bars['day'] = bars.index.date
    groups = bars.groupby('day')['Close']
    bars['fast'] = groups.transform(lambda x: x.rolling(5).mean())
    bars['slow'] = groups.transform(lambda x: x.rolling(20).mean())
    bars['trend'] = groups.transform(lambda x: x.rolling(50).mean())
    return bars


def signal_for_day(day, i):
    if i < 50:
        return None
    previous = day.iloc[i - 1]
    current = day.iloc[i]
    needed = [previous.fast, previous.slow, current.fast, current.slow, current.trend]
    if any(pd.isna(x) for x in needed):
        return None
    long_signal = previous.fast <= previous.slow and current.fast > current.slow
    short_signal = previous.fast >= previous.slow and current.fast < current.slow
    if long_signal and current.Close > current.trend:
        return 'LONG_SIGNAL'
    if short_signal and current.Close < current.trend:
        return 'SHORT_SIGNAL'
    return None


def append_signals(rows):
    if not rows:
        return
    columns = ['candle_time_ct', 'signal', 'reference_close', 'source']
    new_file = not SIGNALS.exists()
    with SIGNALS.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main():
    now_utc = datetime.now(timezone.utc)
    now_ct = now_utc.astimezone(TZ)
    print(f'MES PAPER SIGNAL MONITOR | {now_ct.isoformat()}')
    print('NO BROKER ORDERS. Yahoo data may be delayed.')

    if now_ct.weekday() >= 5 or not SESSION_START <= now_ct.time() <= SESSION_END:
        print('Outside selected US regular-hours monitoring window; no action.')
        return

    bars = get_closed_session_bars(download_bars(), now_utc)
    if bars.empty:
        print('No closed regular-session bars; no action.')
        return

    latest = bars.index[-1].to_pydatetime()
    age = (now_utc - latest.astimezone(timezone.utc)).total_seconds() / 60
    print(f'Latest closed-bar start (CT): {latest.isoformat()} | age: {age:.1f} min')
    if age > MAX_STALE_MINUTES or age < 0:
        print('STALE OR INVALID FEED: refusing to generate signals.')
        return

    state = load_state()
    last_str = state.get('last_candle_ct')
    if last_str is None:
        # Avoid retroactive trade signals on setup day.
        state['last_candle_ct'] = latest.isoformat()
        state['mode'] = 'PAPER_SIGNALS_ONLY'
        save_state(state)
        print('Initialized at latest candle; waiting for future bars.')
        return

    last = datetime.fromisoformat(last_str)
    if last.tzinfo is None:
        raise RuntimeError('Persisted timestamp lost timezone; refusing to proceed.')
    if latest <= last:
        print('No newly closed bars; no duplicate alerts.')
        return

    new_rows = []
    processed = 0
    for day_date, day in bars.groupby('day', sort=True):
        for i in range(50, len(day)):
            t = day.index[i].to_pydatetime()
            if t <= last:
                continue
            processed += 1
            direction = signal_for_day(day, i)
            if direction:
                new_rows.append({
                    'candle_time_ct': t.isoformat(),
                    'signal': direction,
                    'reference_close': round(float(day.iloc[i].Close), 2),
                    'source': 'Yahoo historical 5m delayed; NOT A FILL',
                })

    append_signals(new_rows)
    state['last_candle_ct'] = latest.isoformat()
    state['mode'] = 'PAPER_SIGNALS_ONLY'
    state['signals_total'] = int(state.get('signals_total', 0)) + len(new_rows)
    save_state(state)
    print(f'Processed {processed} newly closed bar(s); recorded {len(new_rows)} signals.')
    for row in new_rows:
        print('PAPER SIGNAL:', row)
    print('State saved. This is not real-time execution and has no P&L.')


if __name__ == '__main__':
    main()

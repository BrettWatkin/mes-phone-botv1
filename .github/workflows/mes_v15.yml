"""MES V16 delayed-data paper simulator with notifications.

Never connects to a broker. All fills are hypothetical and reconstructed from
Yahoo Finance five-minute OHLC data; orders are NOT placed in any market.
"""
import csv
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

TZ = ZoneInfo('America/Chicago')
STATE_PATH = Path('mes_v15_state.json')
TRADES_PATH = Path('mes_v15_trades.csv')
EVENTS_PATH = Path('.mes_v16_events.json')  # ephemeral, untracked
START = 10000.0
STOP_POINTS = 4.0
TARGET_POINTS = 8.0
POINT_VALUE = 5.0
FEE_ROUND_TRIP = 2.50
SLIPPAGE_POINTS = 0.25
DAILY_LIMIT = 60.0
MAX_AGE_MINUTES = 45
LAST_SESSION_BAR = time(14, 55)
FIELDNAMES = ['entry_time_ct','exit_time_ct','side','entry','exit','exit_reason','pnl_usd','balance_usd']


def get_bars(now):
    raw = yf.download('MES=F', period='5d', interval='5m',
                      auto_adjust=False, progress=False, threads=False,
                      multi_level_index=False)
    if raw is None or raw.empty:
        raise RuntimeError('No MES bars returned; paper simulation halted.')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[['Open','High','Low','Close']].dropna().sort_index()
    if raw.index.tz is None:
        raise RuntimeError('Bars lack timezone; refusing to guess.')
    raw.index = raw.index.tz_convert(TZ)
    raw = raw[~raw.index.duplicated(keep='last')]
    # Require six minutes since bar start, to avoid incomplete candles.
    cutoff = pd.Timestamp(now - timedelta(minutes=6)).tz_convert(TZ)
    raw = raw[raw.index <= cutoff]
    raw = raw.between_time('08:30','14:55')
    return raw[raw.index.dayofweek < 5].copy()


def save_events(events):
    # Only newly generated alerts; never commit this file to GitHub.
    EVENTS_PATH.write_text(json.dumps(events, indent=2) + '\n', encoding='utf-8')


def save(state, new_trades):
    # Append trades before advancing state. GitHub commits files together;
    # a failed job won't commit these changes.
    if new_trades:
        with TRADES_PATH.open('a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if f.tell() == 0:
                w.writeheader()
            w.writerows(new_trades)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def settle(state, pos, raw_exit, when, why, new_trades, events):
    d = pos['direction']
    exit_fill = round(float(raw_exit) - d * SLIPPAGE_POINTS, 2)
    pnl = round(d * (exit_fill - pos['entry']) * POINT_VALUE - FEE_ROUND_TRIP, 2)
    state['balance'] = round(state['balance'] + pnl, 2)
    state['daily_pnl'] = round(state['daily_pnl'] + pnl, 2)
    state['trade_count'] += 1
    row = {
        'entry_time_ct': pos['entry_time_ct'],
        'exit_time_ct': when.isoformat(),
        'side': 'LONG' if d == 1 else 'SHORT',
        'entry': pos['entry'], 'exit': exit_fill,
        'exit_reason': why, 'pnl_usd': pnl,
        'balance_usd': state['balance']
    }
    new_trades.append(row)
    events.append({
        'type': 'CLOSE',
        'message': (f"[PAPER / DELAYED] CLOSE {row['side']} @ {row['exit']:.2f} "
                    f"| {why} | P&L ${pnl:+.2f} | balance ${state['balance']:.2f} "
                    f"| {when.isoformat()}")
    })
    state['position'] = None
    print('PAPER CLOSE:', row)


def main():
    events = []
    # Reset current-run alerts before doing anything else.
    save_events(events)
    now = datetime.now(timezone.utc)
    now_ct = now.astimezone(TZ)
    print(f'MES V16 DELAYED PAPER SIMULATOR | {now_ct.isoformat()}')
    print('NO BROKER. NO LIVE ORDERS. Yahoo candles may be delayed or revised.')
    if now_ct.weekday() >= 5 or not time(8, 30) <= now_ct.time() <= time(15, 20):
        print('Outside monitoring window; no action.')
        return
    bars = get_bars(now)
    if bars.empty:
        print('No closed bars. No action.')
        return
    last_index = bars.index[-1]
    age = (now - last_index.to_pydatetime().astimezone(timezone.utc)).total_seconds()/60
    print(f'Latest bar start: {last_index.isoformat()}, age {age:.1f} min')
    if not 0 <= age <= MAX_AGE_MINUTES:
        print('Data stale; no change to state.')
        return
    groups = bars.groupby(bars.index.date, sort=True)
    bars['fast'] = groups['Close'].transform(lambda x: x.rolling(5).mean())
    bars['slow'] = groups['Close'].transform(lambda x: x.rolling(20).mean())
    bars['trend'] = groups['Close'].transform(lambda x: x.rolling(50).mean())
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    else:
        state = {
            'mode': 'SIMULATED_NO_BROKER', 'balance': START,
            'last_candle_ct': last_index.isoformat(), 'daily_date': str(last_index.date()),
            'daily_pnl': 0.0, 'position': None, 'pending_direction': None,
            'trade_count': 0
        }
        save(state, [])
        print('Initialized flat at latest candle; no retroactive paper trades.')
        return
    last_processed = datetime.fromisoformat(state['last_candle_ct'])
    if last_processed.tzinfo is None:
        raise RuntimeError('Missing timezone in persisted state.')
    if last_index <= last_processed:
        print('No new closed bars.')
        return
    new = bars[bars.index > last_processed]
    new_trades = []
    previous_time = pd.Timestamp(last_processed)
    for tstamp, current in new.iterrows():
        t = tstamp.to_pydatetime()
        day = str(t.date())
        same_day = (previous_time.date() == t.date())
        delta_min = (tstamp - previous_time).total_seconds()/60
        if (same_day and delta_min > 5.1) or (not same_day):
            # Paper positions are not assumed filled across missing bars or sessions.
            # Explicitly cancel any unobserved paper position rather than make up P&L.
            if state.get('position') is not None:
                print('UNCERTAIN FILL / SESSION GAP: discard hypothetical open position without PnL')
                events.append({
                    'type': 'WARNING',
                    'message': ('[PAPER / DELAYED] WARNING: missing session bars; '
                                'discarded a hypothetical position without assigning P&L. '
                                'Review the GitHub log.')
                })
            state['position'] = None
            state['pending_direction'] = None
        if state['daily_date'] != day:
            state['daily_date'] = day
            state['daily_pnl'] = 0.0
        pos = state['position']
        pending = state['pending_direction']
        state['pending_direction'] = None
        just_entered = False
        if pos is None and pending and t.time() < LAST_SESSION_BAR:
            d = int(pending)
            raw_entry = float(current.Open)
            risk = (STOP_POINTS + 2*SLIPPAGE_POINTS)*POINT_VALUE + FEE_ROUND_TRIP
            if state['daily_pnl'] - risk >= -DAILY_LIMIT:
                pos = {'direction': d, 'entry': round(raw_entry + d*SLIPPAGE_POINTS, 2),
                       'entry_time_ct': t.isoformat()}
                state['position'] = pos
                just_entered = True
                print('PAPER OPEN:', 'LONG' if d == 1 else 'SHORT', pos['entry'])
                events.append({
                    'type': 'OPEN',
                    'message': (f"[PAPER / DELAYED] OPEN {'LONG' if d == 1 else 'SHORT'} "
                                f"@ {pos['entry']:.2f} | theoretical stop {pos['entry'] - d*STOP_POINTS:.2f} "
                                f"| target {pos['entry'] + d*TARGET_POINTS:.2f} "
                                f"| {t.isoformat()}")
                })
            else:
                print('Risk budget blocks paper entry.')
                if state.get('last_risk_alert_day') != day:
                    events.append({
                        'type': 'RISK',
                        'message': (f'[PAPER / DELAYED] RISK RESTRICTION: '
                                    f'blocked new simulated MES entry on {day}; '
                                    f'daily realized P&L ${state["daily_pnl"]:+.2f}.')
                    })
                    state['last_risk_alert_day'] = day
        closed_this_bar = False
        if pos is not None:
            d = pos['direction']
            stop = pos['entry'] - d * STOP_POINTS
            target = pos['entry'] + d * TARGET_POINTS
            raw_exit = None
            why = None
            if not just_entered:
                if (d == 1 and current.Open <= stop) or (d == -1 and current.Open >= stop):
                    raw_exit, why = float(current.Open), 'GAP_STOP'
                elif (d == 1 and current.Open >= target) or (d == -1 and current.Open <= target):
                    raw_exit, why = float(current.Open), 'GAP_TARGET'
            if raw_exit is None:
                stop_hit = current.Low <= stop if d == 1 else current.High >= stop
                target_hit = current.High >= target if d == 1 else current.Low <= target
                if stop_hit:
                    raw_exit, why = float(stop), 'STOP'
                elif target_hit:
                    raw_exit, why = float(target), 'TARGET'
            if raw_exit is None and t.time() >= LAST_SESSION_BAR:
                raw_exit, why = float(current.Close), 'SESSION_CLOSE'
            if raw_exit is not None:
                settle(state, pos, raw_exit, t, why, new_trades, events)
                closed_this_bar = True
        if state['position'] is None and not just_entered and not closed_this_bar and t.time() < time(14, 50):
            session = bars[bars.index.date == t.date()]
            loc = session.index.get_loc(tstamp)
            if loc >= 50:
                prev = session.iloc[loc-1]
                curr = session.iloc[loc]
                up = prev.fast <= prev.slow and curr.fast > curr.slow and curr.Close > curr.trend
                down = prev.fast >= prev.slow and curr.fast < curr.slow and curr.Close < curr.trend
                if up or down:
                    state['pending_direction'] = 1 if up else -1
                    print('PAPER SIGNAL:', 'LONG' if up else 'SHORT', 'entry next bar only')
        previous_time = tstamp
        state['last_candle_ct'] = t.isoformat()
    save(state, new_trades)
    save_events(events)
    print(f"Processed {len(new)} candles; hypothetical balance: ${state['balance']:.2f}; daily realized PnL: ${state['daily_pnl']:.2f}; trade count: {state['trade_count']}")
    print('Simulated only. No actual position exists.')


# --- Discord delivery is deliberately separate from the market-data code. ---
# It is only invoked by GitHub Actions *after* the state/trades commit step.
# A secret DISCORD_WEBHOOK is supplied as an environment variable by Actions.
def discord_post(message):
    import os
    import urllib.request
    url = os.environ.get('DISCORD_WEBHOOK', '').strip()
    if not url:
        raise SystemExit('DISCORD_WEBHOOK secret missing. Add it in repository Actions secrets.')
    if not url.startswith('https://discord.com/api/webhooks/'):
        raise SystemExit('Discord webhook secret has an unexpected URL format.')
    payload = json.dumps({
        'username': 'MES Paper Bot',
        'content': message[:1900],
        'allowed_mentions': {'parse': []}
    }).encode('utf-8')
    req = urllib.request.Request(url, payload, {
        'Content-Type': 'application/json',
        'User-Agent': 'MES-Paper-Bot/16'
    }, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            if response.status not in (200, 204):
                raise ValueError('Unexpected response status')
    except Exception:
        # A webhook URL is a secret, so never print a raw network exception.
        raise SystemExit('Discord notification failed. Check the private webhook secret.')
    print('Discord notification delivered (no credentials printed).')


def send_events():
    if not EVENTS_PATH.exists():
        print('No current-run alert file.')
        return
    events = json.loads(EVENTS_PATH.read_text(encoding='utf-8'))
    if not events:
        print('No new simulated trade events. No Discord alert sent.')
        return
    messages = [e['message'] for e in events]
    # Group all events for the run into one notification, bounded by Discord's limit.
    chunk = '[MES V16] Simulated events from latest completed candles:\n'
    sent = 0
    for message in messages:
        text = '\n' + message
        if len(chunk) + len(text) > 1850:
            discord_post(chunk)
            sent += 1
            chunk = '[MES V16] Additional simulated events:\n'
        chunk += text
    if chunk.strip():
        discord_post(chunk)
        sent += 1
    print(f'Posted {len(messages)} simulated events in {sent} Discord notification(s).')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == '--notify':
        send_events()
    elif len(sys.argv) == 2 and sys.argv[1] == '--test-notification':
        discord_post('[MES V16 / TEST] Notification setup works. Paper trading only; no broker orders.')
    elif len(sys.argv) == 2 and sys.argv[1] == '--error-notification':
        discord_post('[MES V16 / ERROR] GitHub Actions paper simulator failed. Open the Actions logs; do not rely on this bot for trading.')
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit('Usage: python mes_paper_trader_v16.py [--notify|--test-notification|--error-notification]')

#!/usr/bin/env python3
"""Speed and reliability summary of a render batch (criteria 11/12): results.jsonl wall times + a monitor log.

  python3 harness/eval_summary.py --results data/output/eval1/results.jsonl --monitor data/output/eval1/monitor.log \
      [--mem-total-gb 121.7] [--out data/output/eval1/eval/speed.md]

results.jsonl: one JSON per render, as written by harness/fast_test.py: shot, variant, status, wall_s, time
(the finish time, local), optional pre_s / sampling_s / post_s / steps_s / error.
Monitor log: one line per sample,
  ISO-time MemAvailableGB SwapUsedGB temp,clock_MHz,power_W,throttle_reasons container_status,restarts
e.g. "2026-10-24T10:00:05 82.1 0.0 61,2405,38.5,0x0 running,0". Fields that are missing or not numbers
("N/A", "[N/A]") are skipped. A clip's window is [time - wall_s, time]; samples in it give its peak memory
(= mem-total - min MemAvailable, when --mem-total-gb is given), min available memory, max swap, max temperature,
min clock, max power and the throttle reasons seen (hex masks decoded; 0, the GPU-idle bit, "none" and
"Not Active" count as no throttling).
Time zones are dropped: both files are assumed to use the machine's local time.
Python standard library only.
"""
import argparse, json, re, statistics
from datetime import datetime, timedelta

OK_THROTTLE = {"0", "0x0", "0x0000000000000000", "none", "not active", "active:none", "", "n/a", "[n/a]"}
# nvidia-smi clocks_event_reasons bits; 0x1 (GPU idle) is not a slowdown and is ignored
BITS = {0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal",
        0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks"}


def throttle_name(t):
    """'' when the reasons are benign, else a readable name (hex masks decoded, other text passed through)."""
    t = (t or "").strip()
    if t.lower() in OK_THROTTLE:
        return ""
    if re.fullmatch(r"0x[0-9a-fA-F]+", t):
        v = int(t, 16) & ~0x1
        return "+".join(n for b, n in BITS.items() if v & b) or ("" if v == 0 else t)
    return t


def ptime(s):
    s = s.strip().replace("Z", "+00:00")
    try:
        t = datetime.fromisoformat(s)
    except ValueError:
        t = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    return t.replace(tzinfo=None)


def num(x):
    try:
        v = float(re.sub(r"[^0-9.eE+\-]", "", x))
        return v
    except (ValueError, TypeError):
        return None


def read_monitor(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) < 3 or not p[0][:1].isdigit():
            continue
        try:
            t = ptime(p[0])
        except ValueError:
            continue
        r = {"t": t, "avail": num(p[1]), "swap": num(p[2])}
        if len(p) > 3:
            g = p[3].split(",")
            g += [""] * (4 - len(g))
            r.update(temp=num(g[0]), clock=num(g[1]), power=num(g[2]), throttle=",".join(g[3:]).strip())
        if len(p) > 4:
            c = p[4].split(",")
            r.update(status=c[0], restarts=int(num(c[1])) if len(c) > 1 and num(c[1]) is not None else None)
        rows.append(r)
    return sorted(rows, key=lambda r: r["t"])


def read_results(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        out.append(r)
    return out


def agg(vals, f):
    vals = [v for v in vals if v is not None]
    return f(vals) if vals else None


def fmt(v, k=1):
    return "" if v is None else (f"{v:.{k}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--monitor", default=None)
    ap.add_argument("--mem-total-gb", type=float, default=None, help="MemTotal in GB, to turn MemAvailable into used")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = read_results(a.results)
    if a.variants:
        res = [r for r in res if r.get("variant") in a.variants]
    mon = read_monitor(a.monitor) if a.monitor else []
    md = [f"# Speed and reliability: {a.results}" + (f" + {a.monitor}" if a.monitor else ""), ""]
    md += ["| shot | variant | status | wall s | pre s | sampling s | post s | peak used GB | min avail GB | max swap GB "
           "| max temp C | min clock MHz | max power W | throttle | container |",
           "|" + "---|" * 15]
    per_var = {}
    for r in res:
        v = r.get("variant", "?")
        st = r.get("status", "?")
        w = r.get("wall_s")
        s = {}
        if mon and r.get("time") and w:
            end = ptime(r["time"])
            beg = end - timedelta(seconds=float(w))
            win = [m for m in mon if beg - timedelta(seconds=1) <= m["t"] <= end + timedelta(seconds=1)]
            if win:
                s["min_avail"] = agg([m.get("avail") for m in win], min)
                s["peak_used"] = a.mem_total_gb - s["min_avail"] if a.mem_total_gb and s["min_avail"] is not None else None
                s["max_swap"] = agg([m.get("swap") for m in win], max)
                s["max_temp"] = agg([m.get("temp") for m in win], max)
                s["min_clock"] = agg([m.get("clock") for m in win], min)
                s["max_power"] = agg([m.get("power") for m in win], max)
                th = sorted({throttle_name(m.get("throttle")) for m in win} - {""})
                s["throttle"] = " ".join(th) or "none"
                stat = sorted({m.get("status") for m in win if m.get("status")})
                rs = [m.get("restarts") for m in win if m.get("restarts") is not None]
                s["container"] = ",".join(stat) + (f" restarts+{max(rs) - min(rs)}" if rs and max(rs) > min(rs) else "")
                s["samples"] = len(win)
        pv = per_var.setdefault(v, {"n": 0, "ok": 0, "wall": [], "fail": [], "min_avail": [], "swap": [], "temp": [],
                                    "clock": [], "throttle": set()})
        pv["n"] += 1
        if st == "success":
            pv["ok"] += 1
            if w:
                pv["wall"].append(float(w))
        else:
            pv["fail"].append(f"{r.get('shot')}: {st} {str(r.get('error', ''))[:60]}")
        for k, kk in (("min_avail", "min_avail"), ("max_swap", "swap"), ("max_temp", "temp"), ("min_clock", "clock")):
            if s.get(k) is not None:
                pv[kk].append(s[k])
        if s.get("throttle") and s["throttle"] != "none":
            pv["throttle"].add(s["throttle"])
        md.append(f"| {r.get('shot')} | {v} | {st} | {fmt(w)} | {fmt(r.get('pre_s'))} | {fmt(r.get('sampling_s'))} "
                  f"| {fmt(r.get('post_s'))} | {fmt(s.get('peak_used'))} | {fmt(s.get('min_avail'))} | {fmt(s.get('max_swap'))} "
                  f"| {fmt(s.get('max_temp'), 0)} | {fmt(s.get('min_clock'), 0)} | {fmt(s.get('max_power'))} "
                  f"| {s.get('throttle', '')} | {s.get('container', '')} |")
    md += ["", "## Per variant", "",
           "| variant | renders | failures | wall median s | wall min-max s | wall p90 s | min avail GB | max swap GB "
           "| max temp C | min clock MHz | throttle seen |", "|" + "---|" * 11]
    for v, p in per_var.items():
        wl = sorted(p["wall"])
        p90 = wl[min(len(wl) - 1, int(round(0.9 * (len(wl) - 1))))] if wl else None
        md.append(f"| {v} | {p['n']} | {p['n'] - p['ok']} | {fmt(statistics.median(wl) if wl else None)} "
                  f"| {fmt(wl[0] if wl else None)}-{fmt(wl[-1] if wl else None)} | {fmt(p90)} "
                  f"| {fmt(agg(p['min_avail'], min))} | {fmt(agg(p['swap'], max))} | {fmt(agg(p['temp'], max), 0)} "
                  f"| {fmt(agg(p['clock'], min), 0)} | {' '.join(sorted(p['throttle'])) or 'none'} |")
    fails = [f for p in per_var.values() for f in p["fail"]]
    md += ["", "## Failures", ""] + ([f"- {f}" for f in fails] or ["none"])
    if mon:
        rs = [m.get("restarts") for m in mon if m.get("restarts") is not None]
        md += ["", "## Whole monitor log", "",
               f"- {len(mon)} samples, {mon[0]['t']} to {mon[-1]['t']}",
               f"- min MemAvailable {fmt(agg([m.get('avail') for m in mon], min))} GB, max swap used "
               f"{fmt(agg([m.get('swap') for m in mon], max))} GB",
               f"- max temp {fmt(agg([m.get('temp') for m in mon], max), 0)} C, min clock "
               f"{fmt(agg([m.get('clock') for m in mon], min), 0)} MHz, max power {fmt(agg([m.get('power') for m in mon], max))} W",
               f"- throttle reasons seen: {' '.join(sorted({throttle_name(m.get('throttle')) for m in mon} - {''})) or 'none'}",
               f"- container states: {', '.join(sorted({m.get('status') for m in mon if m.get('status')}))}"
               + (f"; restarts {min(rs)} -> {max(rs)}" if rs else "")]
    text = "\n".join(md) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()

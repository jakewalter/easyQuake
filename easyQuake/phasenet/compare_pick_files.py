#!/usr/bin/env python3
"""Compare two PhaseNet pick files (CSV/TSV) and report differences.

Usage:
  python phasenet/compare_pick_files.py tf1.csv tf2.csv [--tol 0.2] [--dt 0.01]

The script will try to read common columns: file_name, station_id, phase_time (ISO) or
phase_index + t0 (if phase_time not present). It matches picks by station and phase type
(P/S) within a time tolerance (seconds) and prints a summary and lists unmatched picks.
"""
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def load_picks(path, dt=0.01):
    # Try reading as CSV/TSV with pandas
    try:
        df = pd.read_csv(path, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(path)

    # Normalize column names to lower
    df.columns = [c.strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    # Determine phase_time
    if 'phase_time' in cols:
        df['phase_time_dt'] = pd.to_datetime(df[cols['phase_time']])
    elif 't0' in cols and 'phase_index' in cols:
        # compute from t0 + phase_index*dt
        t0col = cols['t0']
        idxcol = cols['phase_index']
        df['phase_time_dt'] = pd.to_datetime(df[t0col]) + df[idxcol].astype(float).apply(lambda x: timedelta(seconds=x*dt))
    else:
        # try columns 't0' and 'p_idx' lists (legacy format), fallback to row index
        if 't0' in cols and 'p_idx' in cols:
            t0col = cols['t0']
            def expand_rows(r):
                # p_idx stored like [12,34]
                s = r[cols['p_idx']]
                try:
                    # Evaluate safely
                    lst = eval(s) if isinstance(s, str) else list(s)
                except Exception:
                    lst = []
                out = []
                for i in lst:
                    out.append(pd.Series({'file_name': r.get(cols.get('fname', 'file_name'), None),
                                           'station_id': r.get(cols.get('station_id', 'station_id'), None),
                                           'phase_type': 'P',
                                           'phase_time_dt': pd.to_datetime(r[t0col]) + timedelta(seconds=float(i)*dt),
                                           'phase_prob': None}))
                return out
            rows = []
            for _, r in df.iterrows():
                rows.extend(expand_rows(r))
            if len(rows) == 0:
                raise ValueError(f"Could not determine phase_time for file {path}")
            df = pd.DataFrame(rows)
        else:
            raise ValueError(f"Could not determine phase_time column in {path}. Columns: {list(df.columns)}")

    # station id
    if 'station_id' in cols:
        df['station_id_str'] = df[cols['station_id']].astype(str)
    elif 'station' in cols:
        df['station_id_str'] = df[cols['station']].astype(str)
    else:
        # fallback: try fname
        df['station_id_str'] = df.get(cols.get('file_name', df.columns[0]), '').astype(str)

    # phase_type
    if 'phase_type' in cols:
        df['phase_type_norm'] = df[cols['phase_type']].astype(str).str.upper()
    else:
        # try to infer from context (not ideal)
        df['phase_type_norm'] = df.get('phase_type', 'P')

    # phase_prob
    if 'phase_prob' in cols:
        df['phase_prob_num'] = pd.to_numeric(df[cols['phase_prob']], errors='coerce')
    else:
        df['phase_prob_num'] = None

    return df[['file_name' if 'file_name' in df.columns else df.columns[0], 'station_id_str', 'phase_type_norm', 'phase_time_dt', 'phase_prob_num']].rename(columns={
        df.columns[0]: 'file_name'
    })


def match_picks(df_ref, df_cmp, tol=0.2):
    """Match picks in df_cmp to df_ref. Return masks of matched entries."""
    ref_times = df_ref['phase_time_dt'].values
    ref_sta = df_ref['station_id_str'].values
    ref_type = df_ref['phase_type_norm'].values

    cmp_times = df_cmp['phase_time_dt'].values
    cmp_sta = df_cmp['station_id_str'].values
    cmp_type = df_cmp['phase_type_norm'].values

    matched_ref = np.zeros(len(df_ref), dtype=bool)
    matched_cmp = np.zeros(len(df_cmp), dtype=bool)

    for i in range(len(df_ref)):
        # Find candidate cmp picks with same station and type
        candidates = [j for j in range(len(df_cmp)) if (not matched_cmp[j]) and (cmp_sta[j]==ref_sta[i]) and (cmp_type[j]==ref_type[i])]
        if not candidates:
            continue
        # compute time diffs
        diffs = np.array([abs((cmp_times[j] - ref_times[i]).total_seconds()) for j in candidates])
        best = np.argmin(diffs)
        if diffs[best] <= tol:
            j = candidates[best]
            matched_ref[i] = True
            matched_cmp[j] = True
    return matched_ref, matched_cmp


def main():
    p = argparse.ArgumentParser(description='Compare two PhaseNet pick files')
    p.add_argument('ref', help='Reference pick file (e.g., TF1 output)')
    p.add_argument('cmp', help='Comparison pick file (e.g., TF2 output)')
    p.add_argument('--tol', type=float, default=0.2, help='time tolerance in seconds for matching')
    p.add_argument('--dt', type=float, default=0.01, help='sampling dt used to convert indices to time')
    args = p.parse_args()

    df_ref = load_picks(args.ref, dt=args.dt)
    df_cmp = load_picks(args.cmp, dt=args.dt)

    print(f"Reference picks: {len(df_ref)}; Comparison picks: {len(df_cmp)}")

    matched_ref, matched_cmp = match_picks(df_ref, df_cmp, tol=args.tol)

    ref_unmatched = df_ref[~matched_ref]
    cmp_unmatched = df_cmp[~matched_cmp]

    print(f"Matched reference picks: {matched_ref.sum()} / {len(df_ref)}")
    print(f"Matched comparison picks: {matched_cmp.sum()} / {len(df_cmp)}")

    if len(ref_unmatched) > 0:
        print('\nPicks in reference but not in comparison (first 50):')
        print(ref_unmatched[['file_name','station_id_str','phase_type_norm','phase_time_dt','phase_prob_num']].head(50).to_string(index=False))

    if len(cmp_unmatched) > 0:
        print('\nPicks in comparison but not in reference (first 50):')
        print(cmp_unmatched[['file_name','station_id_str','phase_type_norm','phase_time_dt','phase_prob_num']].head(50).to_string(index=False))

    # Save unmatched lists
    ref_unmatched.to_csv('unmatched_ref.csv', index=False)
    cmp_unmatched.to_csv('unmatched_cmp.csv', index=False)
    print('\nSaved unmatched lists to unmatched_ref.csv and unmatched_cmp.csv')

if __name__ == '__main__':
    main()

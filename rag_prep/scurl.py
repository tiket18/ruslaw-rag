#!/usr/bin/env python3
"""
Smart curl wrapper with proxy fallback.
Tries direct first, then via SOCKS5 proxy (1089) if blocked.
"""
import sys
import subprocess
import os

def run_curl(args, use_proxy=False):
    cmd = ['curl', '-s', '-L', '--max-time', '60']
    if use_proxy:
        cmd += ['--socks5-hostname', '127.0.0.1:1089']
    cmd += args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

def main():
    if len(sys.argv) < 2:
        print("Usage: scurl.py <curl_args...>")
        sys.exit(1)
    
    args = sys.argv[1:]
    
    # Try direct first
    result = run_curl(args, use_proxy=False)
    
    # If failed (timeout, connection refused, 403, 500), try proxy
    if result.returncode != 0 or any(x in result.stderr for x in ['timeout', 'refused', 'reset', '403', '500', '502', '503', '504']):
        print(f"Direct failed (code {result.returncode}), trying proxy...", file=sys.stderr)
        result = run_curl(args, use_proxy=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
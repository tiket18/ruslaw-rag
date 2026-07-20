#!/usr/bin/env python3
"""
Memory guard - pre-flight check before heavy operations.
Stops ollama/llama-server if free RAM < threshold.
"""
import os
import sys
import subprocess
import json
from pathlib import Path

MIN_FREE_GB = 8  # minimum free RAM for heavy ops
MIN_FREE_PERCENT = 15

def get_mem_info():
    with open('/proc/meminfo') as f:
        info = {}
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                info[k.strip()] = int(v.split()[0])  # kB
    return info

def check_memory():
    mem = get_mem_info()
    total = mem['MemTotal'] / 1024 / 1024  # GB
    available = mem.get('MemAvailable', mem.get('MemFree', 0)) / 1024 / 1024
    percent = (available / total) * 100
    
    print(f"Total: {total:.1f} GB, Available: {available:.1f} GB ({percent:.1f}%)")
    
    if available < MIN_FREE_GB or percent < MIN_FREE_PERCENT:
        return False, f"Low memory: {available:.1f} GB available ({percent:.1f}%)"
    return True, "OK"

def stop_services():
    """Stop ollama and llama-server processes."""
    print("Stopping heavy services...")
    subprocess.run(['systemctl', 'stop', 'ollama'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'llama-server'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'llama.cpp'], capture_output=True)
    import time
    time.sleep(2)

def start_ollama():
    subprocess.run(['systemctl', 'start', 'ollama'], capture_output=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: memguard.py [check|stop|start|ensure]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == 'check':
        ok, msg = check_memory()
        print(msg)
        sys.exit(0 if ok else 1)
    
    elif cmd == 'stop':
        stop_services()
    
    elif cmd == 'start':
        start_ollama()
    
    elif cmd == 'ensure':
        ok, msg = check_memory()
        if not ok:
            print(f"⚠ {msg}, stopping services...")
            stop_services()
            ok, msg = check_memory()
            if not ok:
                print(f"✗ Still low: {msg}")
                sys.exit(1)
        print(f"✓ {msg}")
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == '__main__':
    main()
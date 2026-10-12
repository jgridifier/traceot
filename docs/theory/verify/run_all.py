"""Run every verification script and write verify/RESULTS.txt (full log + PASS/FAIL table)."""
import os, re, subprocess, sys, time, platform
import numpy, scipy

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["v1_legendre.py", "v2_shift.py", "v3_frechet.py", "v4_dro.py"]
log, rows = [], []
t0 = time.time()
for sc in SCRIPTS:
    t = time.time()
    out = subprocess.run([sys.executable, os.path.join(HERE, sc)], cwd=HERE, capture_output=True, text=True)
    log.append(f"===== {sc} (exit {out.returncode}, {time.time()-t:.1f}s) =====\n{out.stdout}{out.stderr}")
    for line in out.stdout.splitlines():
        m = re.match(r"\[(PASS|FAIL)\] (\S+?):(\S+) +(.*)", line)
        if m: rows.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    if out.returncode != 0:
        rows.append(("FAIL", sc, "crash", f"exit code {out.returncode}"))
npass = sum(r[0] == "PASS" for r in rows); nfail = sum(r[0] == "FAIL" for r in rows)
hdr = (f"TRACE theory verification\nrun: {time.strftime('%Y-%m-%d %H:%M %Z')}, python {platform.python_version()}, numpy {numpy.__version__}, scipy {scipy.__version__}\n"
       f"fixed seeds inside each script; wall time {time.time()-t0:.0f}s\n\nSUMMARY: {npass} PASS, {nfail} FAIL\n\n")
table = "status | check\n" + "\n".join(f"{r[0]:4} | {r[1]}:{r[2]:6} | {r[3].split('  ')[0]}" for r in rows)
with open(os.path.join(HERE, "RESULTS.txt"), "w") as fh:
    fh.write(hdr + table + "\n\nFULL LOG\n" + "\n".join(log))
print(hdr + table)
sys.exit(1 if nfail else 0)

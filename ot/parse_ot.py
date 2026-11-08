import re, sys
txt = open(sys.argv[1]).read()
paths = txt.split('Startpoint')[1:]
best = {}
for p in paths:
    ep = re.search(r'Endpoint\s*:\s*(\S+)', p).group(1)
    rows = re.findall(r'^\s+(pin|port)\s+(-?[\d.]+)\s+(-?[\d.]+)', p, re.M)
    t0 = float(rows[0][2]); arr = float(re.search(r'(-?[\d.]+)\s+data arrival time', p).group(1))
    m = re.search(r'(-?[\d.]+)\s+library setup time', p) or re.search(r'setup\s+(-?[\d.]+)', p)
    su = float(m.group(1)) if m else 0.0
    d = arr - t0
    if ep not in best or d > best[ep][0]: best[ep] = (d, su)
for ep, (d, su) in sorted(best.items(), key=lambda x: -x[1][0])[:8]: print(ep, round(d, 2), su)
import json; json.dump(best, open(sys.argv[1] + '.json', 'w'))

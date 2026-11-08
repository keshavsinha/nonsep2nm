import sys, json, re; sys.path.insert(0,'.')
import numpy as np
from nsim.liberty import read_liberty
from nsim.design import Design
from nsim.timing import Timer
lib = read_liberty(['lib/SIMPLE_TT.lib','lib/INVBUF_TT.lib','lib/SEQ_TT.lib'])
nm = sys.argv[1]
d = Design(f'syn/{nm}.v', lib)
tm = Timer(d, np.zeros(len(d.nets)), c_w=0.0, r_w=0.0, fo_limit=10**9)
port = {d.find(p): 'p_' + ''.join(c if c.isalnum() else '_' for c in p) for p in d.inputs + d.outputs}
ot = {}
for f in sys.argv[2:]:
    ot.update(json.load(open(f)))
rows = []
for (i, suR, suF, k) in tm.ends:
    net = d.nets[i]
    if k < 0: key = port.get(net)
    else:
        c, _, pins = d.inst[k]; dp = [p for p, n in pins.items() if n == net][0]; key = f'u{k}:{dp}'
    if key in ot:
        rows.append((key, tm.arr[i], ot[key][0], suR, ot[key][1]))
r = np.array([(a, b) for _, a, b, _, _ in rows])
err = (r[:, 0] - r[:, 1]) / r[:, 1] * 100
print(nm, 'endpoints compared', len(rows), 'mean err %.2f%%  median %.2f%%  max |err| %.2f%%' % (err.mean(), np.median(err), np.abs(err).max()))
w = np.argmax(r[:, 1]); print('worst OT %.1f ours %.1f' % (r[w, 1], r[w, 0]))
json.dump(dict(n=len(rows), mean=float(err.mean()), median=float(np.median(err)), maxabs=float(np.abs(err).max()), worst_ot=float(r[w,1]), worst_ours=float(r[w,0])), open(f'ot/{nm}_cmp.json','w'))

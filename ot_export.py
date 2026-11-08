import sys, pickle; sys.path.insert(0,'.')
import numpy as np
from nsim.liberty import read_liberty
from nsim.design import Design
from nsim.timing import Timer
lib = read_liberty(['lib/SIMPLE_TT.lib','lib/INVBUF_TT.lib','lib/SEQ_TT.lib'])
nm = sys.argv[1]
d = Design(f'syn/{nm}.v', lib)
ids = {n: f'n{i}' for i, n in enumerate(d.nets)}
ins = [d.find(p) for p in d.inputs]; outs = [d.find(p) for p in d.outputs]
port = {}
for p in d.inputs + d.outputs:
    port[d.find(p)] = 'p_' + ''.join(c if c.isalnum() else '_' for c in p)
lines = []
pi = [port[n] for n in dict.fromkeys(ins)]; po = [port[n] for n in dict.fromkeys(outs) if n not in set(ins)]
lines.append(f'module top ({", ".join(pi + po)});')
lines += [f'input {p};' for p in pi] + [f'output {p};' for p in po]
nets = [ids[n] for n in d.nets if n not in port]
for n in nets: lines.append(f'wire {n};')
nm_of = lambda n: port.get(n, ids.get(n))
for k, (c, _, pins) in enumerate(d.inst):
    conns = ', '.join(f'.{p}({nm_of(n)})' for p, n in pins.items())
    lines.append(f'{c} u{k} ({conns});')
lines.append('endmodule')
open(f'ot/{nm}.v', 'w').write('\n'.join(lines) + '\n')
clk = port[[n for n in ins if n in d.clock_nets][0]]
T = []
T.append(f'clock {clk} 10000 50')
for p in pi:
    if p == clk: continue
    T += [f'at {p} E R 0', f'at {p} E F 0', f'at {p} L R 0', f'at {p} L F 0',
          f'slew {p} E R 10', f'slew {p} E F 10', f'slew {p} L R 10', f'slew {p} L F 10']
T += [f'slew {clk} E R 10', f'slew {clk} E F 10', f'slew {clk} L R 10', f'slew {clk} L F 10', f'at {clk} E R 0', f'at {clk} L R 0', f'at {clk} E F 5000', f'at {clk} L F 5000']
for p in po: T += [f'rat {p} L R 10000', f'rat {p} L F 10000', f'rat {p} E R 0', f'rat {p} E F 0']
open(f'ot/{nm}.timing', 'w').write('\n'.join(T) + '\n')
# our engine with zero wires and no buffer-tree assumption
tm = Timer(d, np.zeros(len(d.nets)), c_w=0.0, r_w=0.0, fo_limit=10**9)
print('ours Tmin %.2f ps' % tm.Tmin_nom)

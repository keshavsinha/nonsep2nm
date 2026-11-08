"""Minimal Liberty (.lib) reader for ASAP7 NLDM libraries.

Extracts, per cell: area, pin directions and capacitances, output functions,
combinational / clock-to-Q timing tables (cell_rise, cell_fall, rise_transition,
fall_transition), setup constraints, and average leakage power.
"""
import re
import numpy as np

_tok = re.compile(r'"(?:[^"\\]|\\.)*"|[{}();:,]|[^\s{}();:,"]+')


def _tokens(text):
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text = text.replace('\\\n', ' ')
    return _tok.findall(text)


def _parse_group(toks, i):
    """Parse a group body starting after '{'. Returns (dict, i)."""
    node = {'attrs': {}, 'groups': []}
    while i < len(toks):
        t = toks[i]
        if t == '}':
            return node, i + 1
        name = t
        nxt = toks[i + 1]
        if nxt == ':':
            val = toks[i + 2]
            node['attrs'][name] = val.strip('"')
            i += 3
            if i < len(toks) and toks[i] == ';':
                i += 1
            continue
        if nxt == '(':
            j = i + 2
            args = []
            while toks[j] != ')':
                if toks[j] != ',':
                    args.append(toks[j].strip('"'))
                j += 1
            j += 1
            if j < len(toks) and toks[j] == '{':
                sub, j = _parse_group(toks, j + 1)
                sub['name'] = name
                sub['args'] = args
                node['groups'].append(sub)
            else:
                node['attrs'].setdefault(name, args)
                if j < len(toks) and toks[j] == ';':
                    j += 1
            i = j
            continue
        i += 1
    return node, i


def _table(g):
    idx1 = np.array([float(x) for x in g['attrs']['index_1'][0].split(',')])
    idx2 = np.array([float(x) for x in g['attrs']['index_2'][0].split(',')]) \
        if 'index_2' in g['attrs'] else None
    vals = [[float(x) for x in row.split(',')] for row in g['attrs']['values']]
    return idx1, idx2, np.array(vals)


def interp2(tab, slew, load):
    """Bilinear interpolation / linear extrapolation in an NLDM table."""
    i1, i2, v = tab
    if i2 is None or v.shape[0] == 1:
        x = i2 if v.shape[0] == 1 else i1
        return float(np.interp(load if v.shape[0] == 1 else slew, x, v.ravel()))
    s = np.clip(slew, i1[0], i1[-1])
    l = np.clip(load, i2[0], i2[-1] * 1.0)
    a = min(max(np.searchsorted(i1, s) - 1, 0), len(i1) - 2)
    b = min(max(np.searchsorted(i2, l) - 1, 0), len(i2) - 2)
    ts = (s - i1[a]) / (i1[a + 1] - i1[a])
    tl = (l - i2[b]) / (i2[b + 1] - i2[b])
    v00, v01, v10, v11 = v[a, b], v[a, b + 1], v[a + 1, b], v[a + 1, b + 1]
    base = (1 - ts) * ((1 - tl) * v00 + tl * v01) + ts * ((1 - tl) * v10 + tl * v11)
    # linear extrapolation in load beyond the table edge
    if load > i2[-1]:
        slope = ((1 - ts) * (v[a, -1] - v[a, -2]) + ts * (v[a + 1, -1] - v[a + 1, -2])) / (i2[-1] - i2[-2])
        base += slope * (load - i2[-1])
    return float(base)


class Cell:
    def __init__(self, name):
        self.name = name
        self.area = 0.0
        self.inputs = {}      # pin -> cap (fF)
        self.outputs = {}     # pin -> function string
        self.arcs = []        # dicts: from, to, type, tables
        self.setup = {}       # data pin -> rise/fall constraint table
        self.leak = 0.0       # pW
        self.seq = None       # dict for ff / latch
        self.clock_pin = None


def read_liberty(paths):
    cells = {}
    for p in paths:
        with open(p) as f:
            toks = _tokens(f.read())
        # skip "library (name) {"
        i = toks.index('{') + 1
        root, _ = _parse_group(toks, i)
        for g in root['groups']:
            if g['name'] != 'cell':
                continue
            c = Cell(g['args'][0])
            c.area = float(g['attrs'].get('area', 0))
            leaks = [float(lg['attrs']['value']) for lg in g['groups']
                     if lg['name'] == 'leakage_power' and 'when' not in lg['attrs']
                     and lg['attrs'].get('related_pg_pin', 'VDD') == 'VDD']
            c.leak = leaks[0] if leaks else 0.0
            for sg in g['groups']:
                if sg['name'] in ('ff', 'latch'):
                    c.seq = {'kind': sg['name'], 'state': sg['args'], **sg['attrs']}
            for pg in g['groups']:
                if pg['name'] != 'pin':
                    continue
                pin = pg['args'][0]
                d = pg['attrs'].get('direction')
                if d == 'input':
                    c.inputs[pin] = float(pg['attrs'].get('capacitance', 0.5))
                    if pg['attrs'].get('clock') == 'true':
                        c.clock_pin = pin
                elif d == 'output':
                    c.outputs[pin] = pg['attrs'].get('function', '')
                for tg in pg['groups']:
                    if tg['name'] != 'timing':
                        continue
                    ttype = tg['attrs'].get('timing_type', 'combinational')
                    rel = tg['attrs'].get('related_pin')
                    tabs = {sub['name']: _table(sub) for sub in tg['groups']
                            if 'values' in sub['attrs']}
                    if d == 'output' and ttype in ('combinational', 'rising_edge',
                                                   'falling_edge', 'combinational_rise',
                                                   'combinational_fall'):
                        c.arcs.append({'from': rel, 'to': pin, 'type': ttype, 't': tabs, 'sense': tg['attrs'].get('timing_sense', 'non_unate')})
                    elif ttype.startswith('setup'):
                        c.setup[pin] = tabs
            cells[c.name] = c
    return cells

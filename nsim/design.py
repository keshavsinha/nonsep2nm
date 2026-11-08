"""Gate-level netlist loading, logic simulation (switching activity) and
static timing analysis (nominal and vectorised Monte Carlo) on ASAP7 cells."""
import re
import numpy as np
from collections import defaultdict
from .liberty import interp2

_inst_re = re.compile(r'^\s*(\w+)\s+(\\\S+|\w+)\s*\((.*?)\);', re.S | re.M)
_pin_re = re.compile(r'\.(\w+)\(\s*([^()]*?)\s*\)')
_assign_re = re.compile(r'^\s*assign\s+(.+?)\s*=\s*(.+?)\s*;', re.M)


def _clean(n):
    n = n.strip()
    return n[1:].strip() if n.startswith('\\') else n


class Design:
    def __init__(self, vfile, cells, name=None):
        self.name = name or vfile
        self.lib = cells
        txt = open(vfile).read()
        hdr = re.search(r'module\s+\w+\s*\((.*?)\);', txt, re.S)
        self.inputs, self.outputs = [], []
        for kind, lst in (('input', self.inputs), ('output', self.outputs)):
            for m in re.finditer(r'^\s*%s\s+(?:\[[^\]]*\]\s*)?(.+?);' % kind, txt, re.M):
                for nm in m.group(1).split(','):
                    lst.append(_clean(nm))
        # alias resolution from assigns
        parent = {}

        def find(a):
            while parent.get(a, a) != a:
                parent[a] = parent.get(parent[a], parent[a])
                a = parent[a]
            return a
        for m in _assign_re.finditer(txt):
            l, r = _clean(m.group(1)), _clean(m.group(2))
            if "'" in r:
                continue
            ra, rb = find(l), find(r)
            if ra != rb:
                parent[ra] = rb
        self.find = find
        self.inst = []   # list of (cellname, instname, {pin: net})
        for m in _inst_re.finditer(txt):
            cname = m.group(1)
            if cname not in cells:
                continue
            pins = {p: find(_clean(n)) for p, n in _pin_re.findall(m.group(3)) if n.strip()}
            self.inst.append((cname, _clean(m.group(2)), pins))
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        L = self.lib
        self.driver = {}                  # net -> (inst idx, pin)
        self.sinks = defaultdict(list)    # net -> [(inst idx, pin)]
        self.is_seq = np.zeros(len(self.inst), bool)
        for k, (c, _, pins) in enumerate(self.inst):
            cell = L[c]
            self.is_seq[k] = cell.seq is not None and cell.seq['kind'] == 'ff'
            for p, n in pins.items():
                if p in cell.outputs:
                    self.driver[n] = (k, p)
                elif p in cell.inputs:
                    self.sinks[n].append((k, p))
        self.clock_nets = {self.find(n) for n in self.inputs if 'clk' in n.lower()}
        # topological order of combinational cells
        indeg = np.zeros(len(self.inst), int)
        for k, (c, _, pins) in enumerate(self.inst):
            if self.is_seq[k]:
                continue
            cell = L[c]
            for p, n in pins.items():
                if p in cell.inputs and n in self.driver and not self.is_seq[self.driver[n][0]]:
                    indeg[k] += 1
        order = [k for k in range(len(self.inst)) if not self.is_seq[k] and indeg[k] == 0]
        head = 0
        while head < len(order):
            k = order[head]; head += 1
            c, _, pins = self.inst[k]
            for p, n in pins.items():
                if p in L[c].outputs:
                    for (s, sp) in self.sinks[n]:
                        if self.is_seq[s]:
                            continue
                        indeg[s] -= 1
                        if indeg[s] == 0:
                            order.append(s)
        ncomb = int((~self.is_seq).sum())
        if len(order) != ncomb:
            # combinational loops (latches) are broken arbitrarily
            rest = [k for k in range(len(self.inst)) if not self.is_seq[k] and k not in set(order)]
            order += rest
        self.order = order
        self.area = np.array([L[c].area for c, _, _ in self.inst])
        self.leak = np.array([L[c].leak for c, _, _ in self.inst])  # pW at TT
        self.nets = sorted(set(self.driver) | set(self.sinks))
        self.net_id = {n: i for i, n in enumerate(self.nets)}

    # ------------------------------------------------------------------
    @staticmethod
    def _py(func, state=None):
        f = func.replace('!', ' ~').replace('*', ' & ').replace('+', ' | ')
        return '(' + f.strip() + ')'

    def simulate_activity(self, cycles=400, width=64, seed=1, held=None):
        """Cycle-based random simulation; returns toggle rate per net
        (transitions per clock cycle)."""
        rng = np.random.default_rng(seed)
        L = self.lib
        held = held or {}
        val = {}
        state = {}
        ones = np.uint64(0xFFFFFFFFFFFFFFFF)
        funcs = {}
        for k, (c, _, pins) in enumerate(self.inst):
            for op, fstr in L[c].outputs.items():
                if not fstr:
                    continue
                expr = self._py(fstr)
                funcs[(k, op)] = compile(expr, '<f>', 'eval')
        toggles = defaultdict(float)
        prev = {}
        for n in self.nets:
            val[n] = np.uint64(0)
        for k in np.where(self.is_seq)[0]:
            state[k] = np.uint64(0)
        for t in range(cycles):
            for n in self.inputs:
                fn = self.find(n)
                if fn in self.clock_nets:
                    continue
                if fn in held:
                    val[fn] = ones if held[fn] else np.uint64(0)
                else:
                    val[fn] = np.uint64(rng.integers(0, 2**63, dtype=np.int64)) * np.uint64(2) + np.uint64(rng.integers(0, 2))
            # flop outputs
            for k in state:
                c, _, pins = self.inst[k]
                cell = L[c]
                sname = cell.seq['state'][0]
                env = {sname: state[k], cell.seq['state'][1]: (~state[k]) & ones}
                for op in cell.outputs:
                    if op in pins:
                        val[pins[op]] = eval(funcs[(k, op)], {}, env) & ones
            # combinational
            for k in self.order:
                c, _, pins = self.inst[k]
                cell = L[c]
                env = {p: val.get(pins.get(p), np.uint64(0)) for p in cell.inputs}
                if cell.seq is not None:  # latch: transparent
                    d = cell.seq.get('data_in', 'D')
                    for op in cell.outputs:
                        if op in pins:
                            v = val.get(pins.get('D'), np.uint64(0))
                            val[pins[op]] = v if 'N' not in op else (~v) & ones
                    continue
                for op in cell.outputs:
                    if op in pins and (k, op) in funcs:
                        val[pins[op]] = eval(funcs[(k, op)], {}, env) & ones
            # next state
            for k in state:
                c, _, pins = self.inst[k]
                cell = L[c]
                env = {p: val.get(pins.get(p), np.uint64(0)) for p in cell.inputs}
                state[k] = eval(compile(self._py(cell.seq['next_state']).strip(), '<n>', 'eval'), {}, env) & ones
            if t > 0:
                for n in self.nets:
                    x = int(val[n] ^ prev[n])
                    toggles[n] += bin(x).count('1')
            prev = dict(val)
        act = np.array([toggles[n] / ((cycles - 1) * width) for n in self.nets])
        return act

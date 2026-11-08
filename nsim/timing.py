"""Nominal NLDM static timing and vectorised Monte Carlo STA (revision 2).

Rise and fall transitions are propagated separately and arc unateness is
respected, as in OpenTimer and OpenSTA. Monte Carlo STA keeps the nominal
slews and loads of each arc and applies a per-instance, per-sample delay
multiplier plus optional clock-latency vectors; the circuit delay of a sample
is the maximum over all timing endpoints of (arrival + setup - capture clock
latency), i.e. the minimum clock period at which that chip instance meets
timing.
"""
import numpy as np
from .liberty import interp2

BUF = 'BUFx4_ASAP7_75t_R'
NEG = -1e30


class Timer:
    def __init__(self, design, hpwl, c_w=0.18, r_w=60.0, in_slew=10.0, clk_slew=10.0,
                 fo_limit=16):
        D, L = design, design.lib
        self.D = D
        nid = D.net_id
        nn = len(D.nets)
        cpin = np.zeros(nn); fanout = np.zeros(nn, int)
        for net, s in D.sinks.items():
            for (k, p) in s:
                cpin[nid[net]] += L[D.inst[k][0]].inputs.get(p, 0.5)
                fanout[nid[net]] += 1
        self.cpin, self.fanout, self.hpwl = cpin, fanout, hpwl
        cw = c_w * hpwl
        load = cpin + cw
        bcell = L[BUF]; cin_b = bcell.inputs['A']
        self.buf_delay = np.zeros(nn)
        hf = fanout > fo_limit
        for i in np.where(hf)[0]:
            lev = int(np.ceil(np.log(fanout[i] / 4.0) / np.log(4.0)))
            per_load = 4 * cin_b + cw[i] / max(lev, 1) / 4
            arc = bcell.arcs[0]['t']
            self.buf_delay[i] = lev * max(interp2(arc['cell_rise'], 20, per_load), interp2(arc['cell_fall'], 20, per_load))
            load[i] = 4 * cin_b + cw[i] / 4
        self.load = load
        self.wire_nom = 0.69 * r_w * hpwl * (cw / 2 + cpin) * 1e-3
        for i in np.where(hf)[0]:
            lev = int(np.ceil(np.log(fanout[i] / 4.0) / np.log(4.0)))
            wsum = 0.0
            for l in range(1, lev + 1):
                Ls = hpwl[i] / 2 ** l
                wsum += 0.69 * r_w * Ls * (c_w * Ls / 2 + 4 * cin_b) * 1e-3
            self.wire_nom[i] = wsum
        self.in_slew, self.clk_slew = in_slew, clk_slew
        self._nominal()

    def _nominal(self):
        D, L = self.D, self.D.lib
        nid = D.net_id
        nn = len(D.nets)
        aR = np.full(nn, NEG); aF = np.full(nn, NEG)
        sR = np.full(nn, self.in_slew); sF = np.full(nn, self.in_slew)
        self.pred = {}
        for n in D.inputs:
            f = D.find(n)
            if f in nid:
                aR[nid[f]] = 0.0; aF[nid[f]] = 0.0
        self.launch = []    # (inst, out id, dR, dF)
        for k in np.where(D.is_seq)[0]:
            c, _, pins = D.inst[k]
            for arc in L[c].arcs:
                if arc['to'] in pins and arc['type'] == 'rising_edge':
                    o = nid[pins[arc['to']]]; t = arc['t']; ld = self.load[o]
                    dR = interp2(t['cell_rise'], self.clk_slew, ld); dF = interp2(t['cell_fall'], self.clk_slew, ld)
                    aR[o] = dR + self.buf_delay[o]; aF[o] = dF + self.buf_delay[o]
                    sR[o] = interp2(t['rise_transition'], self.clk_slew, ld); sF[o] = interp2(t['fall_transition'], self.clk_slew, ld)
                    self.launch.append((k, o, dR, dF))
        # per comb instance: (k, o, [(i, dRR, dFR, dRF, dFF)])  dXY: input transition X -> output transition Y
        self.cell_arcs = []
        for k in D.order:
            c, _, pins = D.inst[k]
            cell = L[c]
            outs = {}
            for arc in cell.arcs:
                if arc['to'] not in pins or arc['from'] not in pins:
                    continue
                i = nid[pins[arc['from']]]; o = nid[pins[arc['to']]]
                t = arc['t']; ld = self.load[o]; sense = arc.get('sense', 'non_unate')
                rr = fr = rf = ff = NEG
                if sense in ('positive_unate', 'non_unate'):
                    rr = interp2(t['cell_rise'], sR[i], ld); ff = interp2(t['cell_fall'], sF[i], ld)
                if sense in ('negative_unate', 'non_unate'):
                    fr = interp2(t['cell_rise'], sF[i], ld); rf = interp2(t['cell_fall'], sR[i], ld)
                outs.setdefault(o, []).append((i, rr, fr, rf, ff, arc, sense))
            for o, lst in outs.items():
                bR = bF = NEG; slR = slF = 0.0
                for (i, rr, fr, rf, ff, arc, sense) in lst:
                    w = self.wire_nom[i]
                    inR = (aR[i] if aR[i] > NEG / 2 else 0.0) + w
                    inF = (aF[i] if aF[i] > NEG / 2 else 0.0) + w
                    cR = max(inR + rr, inF + fr); cF = max(inR + rf, inF + ff)
                    if max(cR, cF) > max(bR, bF):
                        self.pred[o] = (i, k)
                    bR = max(bR, cR); bF = max(bF, cF)
                    t = arc['t']; ld = self.load[o]
                    if sense in ('positive_unate', 'non_unate'):
                        slR = max(slR, interp2(t['rise_transition'], sR[i], ld)); slF = max(slF, interp2(t['fall_transition'], sF[i], ld))
                    if sense in ('negative_unate', 'non_unate'):
                        slR = max(slR, interp2(t['rise_transition'], sF[i], ld)); slF = max(slF, interp2(t['fall_transition'], sR[i], ld))
                aR[o] = bR + self.buf_delay[o]; aF[o] = bF + self.buf_delay[o]
                sR[o] = slR; sF[o] = slF
                self.cell_arcs.append((k, o, [(i, rr, fr, rf, ff) for (i, rr, fr, rf, ff, _, _) in lst]))
            for op in cell.outputs:
                if op in pins and nid[pins[op]] not in outs and aR[nid[pins[op]]] <= NEG / 2:
                    aR[nid[pins[op]]] = 0.0; aF[nid[pins[op]]] = 0.0
        aR[aR <= NEG / 2] = 0.0; aF[aF <= NEG / 2] = 0.0
        self.aR, self.aF, self.sR, self.sF = aR, aF, sR, sF
        self.arr = np.maximum(aR, aF)
        self.ends = []      # (net id, setupR, setupF, inst or -1)
        for k in np.where(D.is_seq)[0]:
            c, _, pins = D.inst[k]
            for dp, tabs in L[c].setup.items():
                if dp in pins:
                    i = nid[pins[dp]]
                    suR = interp2(tabs['rise_constraint'], sR[i], self.clk_slew)
                    suF = interp2(tabs['fall_constraint'], sF[i], self.clk_slew)
                    self.ends.append((i, suR, suF, k))
        for n in D.outputs:
            f = D.find(n)
            if f in nid:
                self.ends.append((nid[f], 0.0, 0.0, -1))
        self.end_ids = np.array([e[0] for e in self.ends])
        req = [max(aR[i] + self.wire_nom[i] + sr, aF[i] + self.wire_nom[i] + sf) for (i, sr, sf, k) in self.ends]
        self.end_req = np.array(req)
        self.Tmin_nom = float(np.max(self.end_req))

    def critical_path(self):
        e = int(np.argmax(self.end_req))
        o = self.end_ids[e]
        path = []
        while o in self.pred:
            i, k = self.pred[o]
            path.append(k)
            o = i
        return path[::-1]

    def mc(self, fcell, fwire=None, fseq=None, setup_scale=None, clk=None, clk_po=None):
        D = self.D
        N = fcell.shape[1]
        AR, AF = {}, {}
        for (k, o, dR, dF) in self.launch:
            f = fcell[k]; b = self.buf_delay[o]
            AR[o] = ((dR + b) * f).astype(np.float32); AF[o] = ((dF + b) * f).astype(np.float32)
            if clk is not None:
                AR[o] = AR[o] + clk[k]; AF[o] = AF[o] + clk[k]
        zero = np.zeros(N, np.float32)
        wn = self.wire_nom
        for (k, o, lst) in self.cell_arcs:
            f = fcell[k]
            bR = bF = None
            for (i, rr, fr, rf, ff) in lst:
                r_in = AR.get(i, zero); f_in = AF.get(i, zero)
                if wn[i] > 0:
                    r_in = r_in + wn[i]; f_in = f_in + wn[i]
                cR = None; cF = None
                if rr > NEG / 2: cR = r_in + rr * f
                if fr > NEG / 2: cR = f_in + fr * f if cR is None else np.maximum(cR, f_in + fr * f)
                if ff > NEG / 2: cF = f_in + ff * f
                if rf > NEG / 2: cF = r_in + rf * f if cF is None else np.maximum(cF, r_in + rf * f)
                if cR is not None: bR = cR if bR is None else np.maximum(bR, cR)
                if cF is not None: bF = cF if bF is None else np.maximum(bF, cF)
            if bR is None: bR = zero
            if bF is None: bF = zero
            if self.buf_delay[o] > 0:
                bR = bR + self.buf_delay[o] * f; bF = bF + self.buf_delay[o] * f
            AR[o] = bR; AF[o] = bF
        out = np.full(N, -np.inf, np.float32)
        for (i, suR, suF, k) in self.ends:
            r_in = AR.get(i, zero) + wn[i]; f_in = AF.get(i, zero) + wn[i]
            sc = fcell[k] if (k >= 0 and setup_scale) else 1.0
            a = np.maximum(r_in + suR * sc, f_in + suF * sc)
            if clk is not None:
                a = a - (clk[k] if k >= 0 else clk_po)
            out = np.maximum(out, a)
        return out

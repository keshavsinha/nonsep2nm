* FO4 inverter delay, ASAP7 RVT
.include asap7_TT.mod
.param vdd=0.7
.temp 25
.subckt inv a y vdd
Np y a vdd vdd pmos_rvt l=21n nfin=3
Nn y a 0 0 nmos_rvt l=21n nfin=3
.ends
vdd vdd 0 {vdd}
vin in 0 pulse(0 {vdd} 20p 10p 10p 200p 400p)
x1 in a vdd inv
x2 a b vdd inv
x3 a b2 vdd inv
x4 a b3 vdd inv
x5 a b4 vdd inv
x6 b c vdd inv
x7 b c2 vdd inv
x8 b c3 vdd inv
x9 b c4 vdd inv
.control
tran 0.1p 800p
meas tran tr trig v(a) val={vdd/2} fall=1 targ v(b) val={vdd/2} rise=1
meas tran tf trig v(a) val={vdd/2} rise=1 targ v(b) val={vdd/2} fall=1
.endc
.end

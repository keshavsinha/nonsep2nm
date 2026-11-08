// 8-round fully pipelined Simon-like 32/64 round datapath (one round per stage)
module simonpipe(input clk, input rst, input [31:0] din, input [127:0] key, output [31:0] dout);
  function [15:0] rotl; input [15:0] v; input integer s; rotl = (v << s) | (v >> (16 - s)); endfunction
  reg [15:0] x [0:8]; reg [15:0] y [0:8];
  integer i;
  always @(posedge clk) begin
    if (rst) begin
      for (i = 0; i <= 8; i = i + 1) begin x[i] <= 16'd0; y[i] <= 16'd0; end
    end else begin
      x[0] <= din[31:16]; y[0] <= din[15:0];
      for (i = 0; i < 8; i = i + 1) begin
        x[i+1] <= (rotl(x[i],1) & rotl(x[i],8)) ^ rotl(x[i],2) ^ y[i] ^ key[16*i +: 16];
        y[i+1] <= x[i];
      end
    end
  end
  assign dout = {x[8], y[8]};
endmodule

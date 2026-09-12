
 module fully_connected (
   input clk,
   input rst_n,
   input valid_in,
   input signed [11:0] data_in_1, data_in_2, data_in_3,
   output reg [11:0] data_out,
   output reg valid_out_fc
 );

 localparam INPUT_NUM = 48;
 localparam OUTPUT_NUM = 10;
 localparam DATA_BITS = 8;
 localparam INPUT_WIDTH = 16;
 localparam INPUT_NUM_DATA_BITS = 5;

 localparam [0:3839] W_FC = {
     384'h06FAF3E40012FEF5181EFD001703EE0E0DDEF12EEDFBEDEA0546F5E9EE12FDDC15FA02F1110AE2F90ED80E2500E9F40D,  // out 0
     384'h121B3F13F8FC090AF9F917280D230FEEF8D90AE2E8CAF8100ACB3AF9ECEFF326DF0F1815FCFD1AF30E09EDE814F404F9,  // out 1
     384'hEAF0F5F3E7F80914EBE20D272014EEC41DE4F80C0DE9F80D27032A0BFF080A1FEBE6D3D200FC150D3F151B1202F0E4E9,  // out 2
     384'hC8C70F0AE4EBFE180FE5F7F0D7CCF90F0703140900F004EB12FE0C1D2F0C0A04E1EFEC00FC311BEA02F8F0F2EE003304,  // out 3
     384'h23322A311D0CFB0BFB0CF308060D180CF408ECF1172A051C00C5F6F0EDD5FCFC01FC132B05E206FDF2D0FBEC0A00E8F3,  // out 4
     384'h1416DBF2121AE4BB00020CFAD3F6150F0025F8D90C0A03162607FB0F0D11160B0607F32E00FAB9DDEBE4DFD9001F210F,  // out 5
     384'h1B34160A1218FBB31324D9E4080CF600F60607C2EDFC0416F94624EBBFF4E4E210131BED0B09F0F603FFE80603D8FA07,  // out 6
     384'hE6D3E800CE070E23FC21FE000711250E08DD0A3307DCFF1FD2C7FCEA15E312FCF1E11635E2F5F90D1E28181C0E0014EA,  // out 7
     384'h0600F7F609FDF9F6D7FFF8FB26FA00041122FDFD04F60ED9E821F8120928030A1BF6E5F2EBF71F0F2707DEE309E2FBFC,  // out 8
     384'hD8D5DAF52209050BF9F60820E5F61C1FE52914162623EDE8FAF0AD060A00FC0B34F1EFD8F0081502C9DB080DF8FC0BF8  // out 9
   };
 localparam [0:79] B_FC = 80'h05F90CF5F621E804F2EE;

 reg state;
 reg [INPUT_WIDTH - 1:0] buf_idx;
 reg [3:0] out_idx;
 reg signed [13:0] buffer [0:INPUT_NUM - 1];
 wire signed [DATA_BITS - 1:0] weight [0:INPUT_NUM * OUTPUT_NUM - 1];
 wire signed [DATA_BITS - 1:0] bias [0:OUTPUT_NUM - 1];

 wire signed [13:0] data1, data2, data3;
 reg valid_issue;
 reg valid_mac;
 reg valid_add;
 reg signed [DATA_BITS - 1:0] wsel_r [0:INPUT_NUM - 1];
 reg signed [DATA_BITS - 1:0] bsel_r;
 // 12 partial sums after 48 -> 24 -> 12, 20-bit wrap.
 reg signed [19:0] mid12 [0:11];
 integer wi;
 integer mj;

genvar gi;
generate
    for(gi=0;gi<=479;gi=gi+1) begin : unpack_weight
        assign weight[gi]=W_FC[(8*gi)+:8];
    end
    for(gi=0;gi<=9;gi=gi+1) begin : unpack_bias
        assign bias[gi]=B_FC[(8*gi)+:8];
    end
endgenerate

 wire [OUTPUT_NUM - 1:0] sel;
 wire signed [DATA_BITS - 1:0] wsel [0:INPUT_NUM - 1];
 wire signed [DATA_BITS - 1:0] bsel;

genvar gd, gk;
generate
    for(gd=0;gd<OUTPUT_NUM;gd=gd+1) begin : decode_out_idx
        assign sel[gd]=(out_idx==gd);
    end

    for(gk=0;gk<INPUT_NUM;gk=gk+1) begin : select_weight
        wire [DATA_BITS * OUTPUT_NUM - 1:0] cand;
        for(gd=0;gd<OUTPUT_NUM;gd=gd+1) begin : gate
            assign cand[DATA_BITS*gd+:DATA_BITS]=
                   {DATA_BITS{sel[gd]}} & W_FC[(8*(gd*INPUT_NUM+gk))+:8];
        end
        assign wsel[gk]=((cand[ 7: 0]|cand[15: 8])|(cand[23:16]|cand[31:24]))
                       |((cand[39:32]|cand[47:40])|(cand[55:48]|cand[63:56]))
                       | (cand[71:64]|cand[79:72]);
    end
endgenerate

 wire [DATA_BITS * OUTPUT_NUM - 1:0] bcand;
generate
    for(gd=0;gd<OUTPUT_NUM;gd=gd+1) begin : select_bias
        assign bcand[DATA_BITS*gd+:DATA_BITS]=
               {DATA_BITS{sel[gd]}} & B_FC[(8*gd)+:8];
    end
endgenerate
 assign bsel=((bcand[ 7: 0]|bcand[15: 8])|(bcand[23:16]|bcand[31:24]))
            |((bcand[39:32]|bcand[47:40])|(bcand[55:48]|bcand[63:56]))
            | (bcand[71:64]|bcand[79:72]);

 assign data1 = (data_in_1[11] == 1) ? {2'b11, data_in_1} : {2'b00, data_in_1};
 assign data2 = (data_in_2[11] == 1) ? {2'b11, data_in_2} : {2'b00, data_in_2};
 assign data3 = (data_in_3[11] == 1) ? {2'b11, data_in_3} : {2'b00, data_in_3};

 always @(posedge clk) begin
   if(~rst_n) begin
     valid_issue <= 0;
     buf_idx <= 0;
     out_idx <= 0;
     state <= 0;
   end else begin
     if(valid_issue == 1) begin
       valid_issue <= 0;
     end

     if(valid_in == 1) begin
       if(!state) begin
         buffer[buf_idx] <= data1;
         buffer[INPUT_WIDTH + buf_idx] <= data2;
         buffer[INPUT_WIDTH * 2 + buf_idx] <= data3;
         buf_idx <= buf_idx + 1'b1;
         if(buf_idx == INPUT_WIDTH - 1) begin
           buf_idx <= 0;
           state <= 1;
           valid_issue <= 1;
         end
       end else begin
         out_idx <= out_idx + 1'b1;
         if(out_idx == OUTPUT_NUM - 1) begin
           out_idx <= 0;
         end
         valid_issue <= 1;
       end
     end
   end
 end

 always @(posedge clk) begin
   for (wi = 0; wi < INPUT_NUM; wi = wi + 1)
     wsel_r[wi] <= wsel[wi];
   bsel_r <= bsel;
 end

 // ---------------------------------------------------------------------
 //  MAC cut at 12 partial sums (option A).
 //
 //  Stage 1 : 48 products + 48->24->12.  Stage 2 : 12->6->3->2->1 + slice.
 //  20-bit wrap at every '+' matches the old calc_out assignment.
 //  valid_issue -> valid_mac -> valid_add -> valid_out_fc
 // ---------------------------------------------------------------------
 wire signed [19:0] prod [0:47];
 wire signed [19:0] lv24 [0:23];
 wire signed [19:0] lv12 [0:11];
 wire signed [19:0] lv6 [0:5];
 wire signed [19:0] lv3 [0:2];
 wire signed [19:0] sum_tot;
 genvar gt;

 generate
   for (gt = 0; gt < INPUT_NUM; gt = gt + 1) begin : MUL
     assign prod[gt] = wsel_r[gt] * buffer[gt];
   end
   for (gt = 0; gt < 23; gt = gt + 1) begin : RED24
     assign lv24[gt] = prod[2*gt] + prod[2*gt+1];
   end
   for (gt = 0; gt < 12; gt = gt + 1) begin : RED12
     assign lv12[gt] = lv24[2*gt] + lv24[2*gt+1];
   end
   for (gt = 0; gt < 6; gt = gt + 1) begin : RED6
     assign lv6[gt] = mid12[2*gt] + mid12[2*gt+1];
   end
 endgenerate
 assign lv24[23] = prod[46] + (prod[47] + bsel_r);
 assign lv3[0] = lv6[0] + lv6[1];
 assign lv3[1] = lv6[2] + lv6[3];
 assign lv3[2] = lv6[4] + lv6[5];
 assign sum_tot = (lv3[0] + lv3[1]) + lv3[2];

 always @(posedge clk) begin
   if(~rst_n) begin
     for (mj = 0; mj < 12; mj = mj + 1)
       mid12[mj] <= 20'd0;
     data_out <= 12'd0;
     valid_mac <= 1'b0;
     valid_add <= 1'b0;
     valid_out_fc <= 1'b0;
   end else begin
     for (mj = 0; mj < 12; mj = mj + 1)
       mid12[mj] <= lv12[mj];
     data_out <= sum_tot[18:7];
     valid_mac <= valid_issue;
     valid_add <= valid_mac;
     valid_out_fc <= valid_add;
   end
 end

 endmodule

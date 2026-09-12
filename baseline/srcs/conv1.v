/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : conv1_layer.v
 *  Written by : Kang, Bo Young
 *  Written on : Sep 30, 2021
 *  Version    : 21.2
 *  Design     : 1st Convolution Layer for CNN MNIST dataset
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: conv1_layer
 *------------------------------------------------------------------*/
 
 module conv1_layer (
   input clk,
   input rst_n,
   input [7:0] data_in,
   output [11:0] conv_out_1, conv_out_2, conv_out_3,
   output valid_out_conv
 );

 wire [7:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4,
  data_out_5, data_out_6, data_out_7, data_out_8, data_out_9,
  data_out_10, data_out_11, data_out_12, data_out_13, data_out_14,
  data_out_15, data_out_16, data_out_17, data_out_18, data_out_19,
  data_out_20, data_out_21, data_out_22, data_out_23, data_out_24;
 wire valid_out_buf;

 conv1_buf #(.WIDTH(28), .HEIGHT(28), .DATA_BITS(8)) conv1_buf(
   .clk(clk),
   .rst_n(rst_n),
   .data_in(data_in),
   .data_out_0(data_out_0),
   .data_out_1(data_out_1),
   .data_out_2(data_out_2),
   .data_out_3(data_out_3),
   .data_out_4(data_out_4),
   .data_out_5(data_out_5),
   .data_out_6(data_out_6),
   .data_out_7(data_out_7),
   .data_out_8(data_out_8),
   .data_out_9(data_out_9),
   .data_out_10(data_out_10),
   .data_out_11(data_out_11),
   .data_out_12(data_out_12),
   .data_out_13(data_out_13),
   .data_out_14(data_out_14),
   .data_out_15(data_out_15),
   .data_out_16(data_out_16),
   .data_out_17(data_out_17),
   .data_out_18(data_out_18),
   .data_out_19(data_out_19),
   .data_out_20(data_out_20),
   .data_out_21(data_out_21),
   .data_out_22(data_out_22),
   .data_out_23(data_out_23),
   .data_out_24(data_out_24),
   .valid_out_buf(valid_out_buf)
 );

 conv1_calc conv1_calc(
   .clk(clk),
   .rst_n(rst_n),
   .valid_out_buf(valid_out_buf),
   .data_out_0(data_out_0),
   .data_out_1(data_out_1),
   .data_out_2(data_out_2),
   .data_out_3(data_out_3),
   .data_out_4(data_out_4),
   .data_out_5(data_out_5),
   .data_out_6(data_out_6),
   .data_out_7(data_out_7),
   .data_out_8(data_out_8),
   .data_out_9(data_out_9),
   .data_out_10(data_out_10),
   .data_out_11(data_out_11),
   .data_out_12(data_out_12),
   .data_out_13(data_out_13),
   .data_out_14(data_out_14),
   .data_out_15(data_out_15),
   .data_out_16(data_out_16),
   .data_out_17(data_out_17),
   .data_out_18(data_out_18),
   .data_out_19(data_out_19),
   .data_out_20(data_out_20),
   .data_out_21(data_out_21),
   .data_out_22(data_out_22),
   .data_out_23(data_out_23),
   .data_out_24(data_out_24),
   .conv_out_1(conv_out_1),
   .conv_out_2(conv_out_2),
   .conv_out_3(conv_out_3),
   .valid_out_calc(valid_out_conv)
 );
 endmodule

/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : conv1_buf.v
 *  Written by : Kang, Bo Young
 *  Written on : Sep 30, 2021
 *  Version    : 21.2
 *  Design     : 1st Convolution Layer for CNN MNIST dataset
 *               Input Buffer
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: conv1_buf
 *------------------------------------------------------------------*/
 
 module conv1_buf #(parameter WIDTH = 28, HEIGHT = 28, DATA_BITS = 8)(
   input clk,
   input rst_n,
   input [DATA_BITS - 1:0] data_in,
   output reg [DATA_BITS - 1:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4,
   data_out_5, data_out_6, data_out_7, data_out_8, data_out_9,
   data_out_10, data_out_11, data_out_12, data_out_13, data_out_14,
   data_out_15, data_out_16, data_out_17, data_out_18, data_out_19,
   data_out_20, data_out_21, data_out_22, data_out_23, data_out_24,
   output reg valid_out_buf
 );

 localparam FILTER_SIZE = 5;

 reg [DATA_BITS - 1:0] buffer [0:WIDTH * FILTER_SIZE - 1];
 reg [DATA_BITS - 1:0] buf_idx;
 reg [4:0] w_idx, h_idx;
 reg [2:0] buf_flag;  // 0 ~ 4
 reg state;

 always @(posedge clk) begin
   if(~rst_n) begin
     buf_idx <= -1;
     w_idx <= 0;
     h_idx <= 0;
     buf_flag <= 0;
     state <= 0;
     valid_out_buf <= 0;
     data_out_0 <= 12'bx;
     data_out_1 <= 12'bx;
     data_out_2 <= 12'bx;
     data_out_3 <= 12'bx;
     data_out_4 <= 12'bx;
     data_out_5 <= 12'bx;
     data_out_6 <= 12'bx;
     data_out_7 <= 12'bx;
     data_out_8 <= 12'bx;
     data_out_9 <= 12'bx;
     data_out_10 <= 12'bx;
     data_out_11 <= 12'bx;
     data_out_12 <= 12'bx;
     data_out_13 <= 12'bx;
     data_out_14 <= 12'bx;
     data_out_15 <= 12'bx;
     data_out_16 <= 12'bx;
     data_out_17 <= 12'bx;
     data_out_18 <= 12'bx;
     data_out_19 <= 12'bx;
     data_out_20 <= 12'bx;
     data_out_21 <= 12'bx;
     data_out_22 <= 12'bx;
     data_out_23 <= 12'bx;
     data_out_24 <= 12'bx;
   end else begin
   buf_idx <= buf_idx + 1;
   if(buf_idx == WIDTH * FILTER_SIZE - 1) begin // buffer size = 140 = 28(w) * 5(h)
     buf_idx <= 0;
   end
   
   buffer[buf_idx] <= data_in;  // data input
   
   // Wait until first 140 input data filled in buffer
   if(!state) begin
     if(buf_idx == WIDTH * FILTER_SIZE - 1) begin
       state <= 1'b1;
     end
   end else begin // valid state
     w_idx <= w_idx + 1'b1; // move right

     if(w_idx == WIDTH - FILTER_SIZE + 1) begin
       valid_out_buf <= 1'b0; // unvalid area
     end else if(w_idx == WIDTH - 1) begin
       buf_flag <= buf_flag + 1'b1;
       if(buf_flag == FILTER_SIZE - 1) begin
         buf_flag <= 0;
       end
       w_idx <= 0;

       if(h_idx == HEIGHT - FILTER_SIZE) begin  // done 1 input read -> 28 * 28
         h_idx <= 0;
         state <= 1'b0;
       end 
       
       h_idx <= h_idx + 1'b1;

     end else if(w_idx == 0) begin
       valid_out_buf <= 1'b1; // start valid area
     end

     // Buffer Selection -> 5 * 5
     if(buf_flag == 3'd0) begin
       data_out_0 <= buffer[w_idx];
       data_out_1 <= buffer[w_idx + 1];
       data_out_2 <= buffer[w_idx + 2];
       data_out_3 <= buffer[w_idx + 3];
       data_out_4 <= buffer[w_idx + 4];

       data_out_5 <= buffer[w_idx + WIDTH];
       data_out_6 <= buffer[w_idx + 1 + WIDTH];
       data_out_7 <= buffer[w_idx + 2 + WIDTH];
       data_out_8 <= buffer[w_idx + 3 + WIDTH];
       data_out_9 <= buffer[w_idx + 4 + WIDTH];

       data_out_10 <= buffer[w_idx + WIDTH * 2];
       data_out_11 <= buffer[w_idx + 1 + WIDTH * 2];
       data_out_12 <= buffer[w_idx + 2 + WIDTH * 2];
       data_out_13 <= buffer[w_idx + 3 + WIDTH * 2];
       data_out_14 <= buffer[w_idx + 4 + WIDTH * 2];

       data_out_15 <= buffer[w_idx + WIDTH * 3];
       data_out_16 <= buffer[w_idx + 1 + WIDTH * 3];
       data_out_17 <= buffer[w_idx + 2 + WIDTH * 3];
       data_out_18 <= buffer[w_idx + 3 + WIDTH * 3];
       data_out_19 <= buffer[w_idx + 4 + WIDTH * 3];

       data_out_20 <= buffer[w_idx + WIDTH * 4];
       data_out_21 <= buffer[w_idx + 1 + WIDTH * 4];
       data_out_22 <= buffer[w_idx + 2 + WIDTH * 4];
       data_out_23 <= buffer[w_idx + 3 + WIDTH * 4];
       data_out_24 <= buffer[w_idx + 4 + WIDTH * 4];
     end else if(buf_flag == 3'd1) begin
       data_out_0 <= buffer[w_idx + WIDTH];
       data_out_1 <= buffer[w_idx + 1 + WIDTH];
       data_out_2 <= buffer[w_idx + 2 + WIDTH];
       data_out_3 <= buffer[w_idx + 3 + WIDTH];
       data_out_4 <= buffer[w_idx + 4 + WIDTH];

       data_out_5 <= buffer[w_idx + WIDTH * 2];
       data_out_6 <= buffer[w_idx + 1 + WIDTH * 2];
       data_out_7 <= buffer[w_idx + 2 + WIDTH * 2];
       data_out_8 <= buffer[w_idx + 3 + WIDTH * 2];
       data_out_9 <= buffer[w_idx + 4 + WIDTH * 2];

       data_out_10 <= buffer[w_idx + WIDTH * 3];
       data_out_11 <= buffer[w_idx + 1 + WIDTH * 3];
       data_out_12 <= buffer[w_idx + 2 + WIDTH * 3];
       data_out_13 <= buffer[w_idx + 3 + WIDTH * 3];
       data_out_14 <= buffer[w_idx + 4 + WIDTH * 3];

       data_out_15 <= buffer[w_idx + WIDTH * 4];
       data_out_16 <= buffer[w_idx + 1 + WIDTH * 4];
       data_out_17 <= buffer[w_idx + 2 + WIDTH * 4];
       data_out_18 <= buffer[w_idx + 3 + WIDTH * 4];
       data_out_19 <= buffer[w_idx + 4 + WIDTH * 4];

       data_out_20 <= buffer[w_idx];
       data_out_21 <= buffer[w_idx + 1];
       data_out_22 <= buffer[w_idx + 2];
       data_out_23 <= buffer[w_idx + 3];
       data_out_24 <= buffer[w_idx + 4];
     end else if(buf_flag == 3'd2) begin
       data_out_0 <= buffer[w_idx + WIDTH * 2];
       data_out_1 <= buffer[w_idx + 1 + WIDTH * 2];
       data_out_2 <= buffer[w_idx + 2 + WIDTH * 2];
       data_out_3 <= buffer[w_idx + 3 + WIDTH * 2];
       data_out_4 <= buffer[w_idx + 4 + WIDTH * 2];

       data_out_5 <= buffer[w_idx + WIDTH * 3];
       data_out_6 <= buffer[w_idx + 1 + WIDTH * 3];
       data_out_7 <= buffer[w_idx + 2 + WIDTH * 3];
       data_out_8 <= buffer[w_idx + 3 + WIDTH * 3];
       data_out_9 <= buffer[w_idx + 4 + WIDTH * 3];

       data_out_10 <= buffer[w_idx + WIDTH * 4];
       data_out_11 <= buffer[w_idx + 1 + WIDTH * 4];
       data_out_12 <= buffer[w_idx + 2 + WIDTH * 4];
       data_out_13 <= buffer[w_idx + 3 + WIDTH * 4];
       data_out_14 <= buffer[w_idx + 4 + WIDTH * 4];

       data_out_15 <= buffer[w_idx];
       data_out_16 <= buffer[w_idx + 1];
       data_out_17 <= buffer[w_idx + 2];
       data_out_18 <= buffer[w_idx + 3];
       data_out_19 <= buffer[w_idx + 4];

       data_out_20 <= buffer[w_idx + WIDTH];
       data_out_21 <= buffer[w_idx + 1 + WIDTH];
       data_out_22 <= buffer[w_idx + 2 + WIDTH];
       data_out_23 <= buffer[w_idx + 3 + WIDTH];
       data_out_24 <= buffer[w_idx + 4 + WIDTH];
     end else if(buf_flag == 3'd3) begin
       data_out_0 <= buffer[w_idx + WIDTH * 3];
       data_out_1 <= buffer[w_idx + 1 + WIDTH * 3];
       data_out_2 <= buffer[w_idx + 2 + WIDTH * 3];
       data_out_3 <= buffer[w_idx + 3 + WIDTH * 3];
       data_out_4 <= buffer[w_idx + 4 + WIDTH * 3];

       data_out_5 <= buffer[w_idx + WIDTH * 4];
       data_out_6 <= buffer[w_idx + 1 + WIDTH * 4];
       data_out_7 <= buffer[w_idx + 2 + WIDTH * 4];
       data_out_8 <= buffer[w_idx + 3 + WIDTH * 4];
       data_out_9 <= buffer[w_idx + 4 + WIDTH * 4];

       data_out_10 <= buffer[w_idx];
       data_out_11 <= buffer[w_idx + 1];
       data_out_12 <= buffer[w_idx + 2];
       data_out_13 <= buffer[w_idx + 3];
       data_out_14 <= buffer[w_idx + 4];

       data_out_15 <= buffer[w_idx + WIDTH];
       data_out_16 <= buffer[w_idx + 1 + WIDTH];
       data_out_17 <= buffer[w_idx + 2 + WIDTH];
       data_out_18 <= buffer[w_idx + 3 + WIDTH];
       data_out_19 <= buffer[w_idx + 4 + WIDTH];

       data_out_20 <= buffer[w_idx + WIDTH * 2];
       data_out_21 <= buffer[w_idx + 1 + WIDTH * 2];
       data_out_22 <= buffer[w_idx + 2 + WIDTH * 2];
       data_out_23 <= buffer[w_idx + 3 + WIDTH * 2];
       data_out_24 <= buffer[w_idx + 4 + WIDTH * 2];      
     end else if(buf_flag == 3'd4) begin
       data_out_0 <= buffer[w_idx + WIDTH * 4];
       data_out_1 <= buffer[w_idx + 1 + WIDTH * 4];
       data_out_2 <= buffer[w_idx + 2 + WIDTH * 4];
       data_out_3 <= buffer[w_idx + 3 + WIDTH * 4];
       data_out_4 <= buffer[w_idx + 4 + WIDTH * 4];

       data_out_5 <= buffer[w_idx];
       data_out_6 <= buffer[w_idx + 1];
       data_out_7 <= buffer[w_idx + 2];
       data_out_8 <= buffer[w_idx + 3];
       data_out_9 <= buffer[w_idx + 4];

       data_out_10 <= buffer[w_idx + WIDTH];
       data_out_11 <= buffer[w_idx + 1 + WIDTH];
       data_out_12 <= buffer[w_idx + 2 + WIDTH];
       data_out_13 <= buffer[w_idx + 3 + WIDTH];
       data_out_14 <= buffer[w_idx + 4 + WIDTH];

       data_out_15 <= buffer[w_idx + WIDTH * 2];
       data_out_16 <= buffer[w_idx + 1 + WIDTH * 2];
       data_out_17 <= buffer[w_idx + 2 + WIDTH * 2];
       data_out_18 <= buffer[w_idx + 3 + WIDTH * 2];
       data_out_19 <= buffer[w_idx + 4 + WIDTH * 2];

       data_out_20 <= buffer[w_idx + WIDTH * 3];
       data_out_21 <= buffer[w_idx + 1 + WIDTH * 3];
       data_out_22 <= buffer[w_idx + 2 + WIDTH * 3];
       data_out_23 <= buffer[w_idx + 3 + WIDTH * 3];
       data_out_24 <= buffer[w_idx + 4 + WIDTH * 3];   
     end
   end
   end
 end
endmodule

/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : conv1_calc.v
 *  Written by : Kang, Bo Young
 *  Written on : Oct 1, 2021
 *  Version    : 21.2
 *  Design     : 1st Convolution Layer for CNN MNIST dataset
 *               Convolution Sum Calculation
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: conv1_calc
 *------------------------------------------------------------------*/
 
 module conv1_calc #(parameter WIDTH = 28, HEIGHT = 28, DATA_BITS = 8)(
   input clk,
   input rst_n,
   input valid_out_buf,
   input [DATA_BITS - 1:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4,
   data_out_5, data_out_6, data_out_7, data_out_8, data_out_9,
   data_out_10, data_out_11, data_out_12, data_out_13, data_out_14,
   data_out_15, data_out_16, data_out_17, data_out_18, data_out_19,
   data_out_20, data_out_21, data_out_22, data_out_23, data_out_24,
   output reg signed [11:0] conv_out_1, conv_out_2, conv_out_3,
   output reg valid_out_calc
 );

 localparam FILTER_SIZE = 5;
 localparam CHANNEL_LEN = 3;

 // Trained weights / biases, hard-wired as constants.
 // Regenerate with gen_weights.py after re-quantizing the model.
 localparam [0:199] W_11 = {
     40'hF9EE0DE9FC,  // row 0
     40'h0D0F16F2F7,  // row 1
     40'hFC0608FFE9,  // row 2
     40'h1101EC05EC,  // row 3
     40'hF7E711FAF4  // row 4
   };
 localparam [0:199] W_12 = {
     40'h00F1295C4F,  // row 0
     40'hE6E01D5047,  // row 1
     40'hE005236021,  // row 2
     40'hFD1A355A06,  // row 3
     40'h171C2D342C  // row 4
   };
 localparam [0:199] W_13 = {
     40'h1D08E8E8E8,  // row 0
     40'h2A2BE3B4B4,  // row 1
     40'h161CF90AFC,  // row 2
     40'h181D3A4D3A,  // row 3
     40'h023945532A  // row 4
   };
 localparam [0:23] B_1 = 24'hEB000C;



 wire signed [19:0] calc_out_1, calc_out_2, calc_out_3;
 wire signed [DATA_BITS:0] exp_data [0:FILTER_SIZE * FILTER_SIZE - 1];
 wire signed [11:0] exp_bias [0:CHANNEL_LEN - 1];

 wire signed [DATA_BITS - 1:0] weight_1 [0:FILTER_SIZE * FILTER_SIZE - 1];
 wire signed [DATA_BITS - 1:0] weight_2 [0:FILTER_SIZE * FILTER_SIZE - 1];
 wire signed [DATA_BITS - 1:0] weight_3 [0:FILTER_SIZE * FILTER_SIZE - 1];
 wire signed [DATA_BITS - 1:0] bias [0:CHANNEL_LEN - 1];

// Constants must be unpacked with continuous assignments, not always @(*):
// with only constants on the right-hand side the implicit sensitivity list is
// empty, so the block would never run and the weights would stay X in sim.
genvar gi;
generate
    for(gi=0;gi<=24;gi=gi+1) begin : unpack_weight
        assign weight_1[gi]=W_11[(8*gi)+:8];
        assign weight_2[gi]=W_12[(8*gi)+:8];
        assign weight_3[gi]=W_13[(8*gi)+:8];
    end
    for(gi=0;gi<=2;gi=gi+1) begin : unpack_bias
        assign bias[gi]=B_1[(8*gi)+:8];
    end
endgenerate

 // Unsigned -> Signed
 assign exp_data[0] = {1'd0, data_out_0};
 assign exp_data[1] = {1'd0, data_out_1};
 assign exp_data[2] = {1'd0, data_out_2};
 assign exp_data[3] = {1'd0, data_out_3};
 assign exp_data[4] = {1'd0, data_out_4};
 assign exp_data[5] = {1'd0, data_out_5};
 assign exp_data[6] = {1'd0, data_out_6};
 assign exp_data[7] = {1'd0, data_out_7};
 assign exp_data[8] = {1'd0, data_out_8};
 assign exp_data[9] = {1'd0, data_out_9};
 assign exp_data[10] = {1'd0, data_out_10};
 assign exp_data[11] = {1'd0, data_out_11};
 assign exp_data[12] = {1'd0, data_out_12};
 assign exp_data[13] = {1'd0, data_out_13};
 assign exp_data[14] = {1'd0, data_out_14};
 assign exp_data[15] = {1'd0, data_out_15};
 assign exp_data[16] = {1'd0, data_out_16};
 assign exp_data[17] = {1'd0, data_out_17};
 assign exp_data[18] = {1'd0, data_out_18};
 assign exp_data[19] = {1'd0, data_out_19};
 assign exp_data[20] = {1'd0, data_out_20};
 assign exp_data[21] = {1'd0, data_out_21};
 assign exp_data[22] = {1'd0, data_out_22};
 assign exp_data[23] = {1'd0, data_out_23};
 assign exp_data[24] = {1'd0, data_out_24};

 //  Re-calibration of extracted weight data according to MSB
 assign exp_bias[0] = (bias[0][7] == 1) ? {4'b1111, bias[0]} : {4'd0, bias[0]};
 assign exp_bias[1] = (bias[1][7] == 1) ? {4'b1111, bias[1]} : {4'd0, bias[1]};
 assign exp_bias[2] = (bias[2][7] == 1) ? {4'b1111, bias[2]} : {4'd0, bias[2]};


 // ---------------------------------------------------------------------
 //  Fused constant-coefficient multiply-accumulate (one tree per channel).
 //
 //  The 25 products used to be a single '+' expression.  Even re-balanced,
 //  that leaves 25 constant-multiplier CPAs plus a carry propagation at
 //  every tree node.  Each multiplier is now expanded into shift/invert
 //  rows and all of them are compressed by a 3:2 carry-save tree; the only
 //  carry propagation left is one Kogge-Stone prefix adder per channel.
 //
 //  Negative weights use |w| instead of the raw two's-complement pattern:
 //  0xF9 (-7) has six set bits but |w| = 7 has three, so half the rows
 //  disappear.  -x == ~x + 1, and all those +1s plus the bias (already
 //  shifted left by 8, so no carry reaches bit 8) fold into one constant
 //  row.
 //
 //  Rows and layer sizes are enumerated at generation time: no parameter
 //  modules and no constant functions for the synthesiser to elaborate.
 //  Regenerate this block together with W_11/W_12/W_13 if the model is
 //  re-quantised.
 // ---------------------------------------------------------------------
localparam NR_CA = 58;
wire [NR_CA*20-1:0] CA00;
assign CA00[0*20 +: 20] = ~{12'd0, data_out_0};
assign CA00[1*20 +: 20] = ~({12'd0, data_out_0} << 1);
assign CA00[2*20 +: 20] = ~({12'd0, data_out_0} << 2);
assign CA00[3*20 +: 20] = ~({12'd0, data_out_1} << 1);
assign CA00[4*20 +: 20] = ~({12'd0, data_out_1} << 4);
assign CA00[5*20 +: 20] = {12'd0, data_out_2};
assign CA00[6*20 +: 20] = ({12'd0, data_out_2} << 2);
assign CA00[7*20 +: 20] = ({12'd0, data_out_2} << 3);
assign CA00[8*20 +: 20] = ~{12'd0, data_out_3};
assign CA00[9*20 +: 20] = ~({12'd0, data_out_3} << 1);
assign CA00[10*20 +: 20] = ~({12'd0, data_out_3} << 2);
assign CA00[11*20 +: 20] = ~({12'd0, data_out_3} << 4);
assign CA00[12*20 +: 20] = ~({12'd0, data_out_4} << 2);
assign CA00[13*20 +: 20] = {12'd0, data_out_5};
assign CA00[14*20 +: 20] = ({12'd0, data_out_5} << 2);
assign CA00[15*20 +: 20] = ({12'd0, data_out_5} << 3);
assign CA00[16*20 +: 20] = {12'd0, data_out_6};
assign CA00[17*20 +: 20] = ({12'd0, data_out_6} << 1);
assign CA00[18*20 +: 20] = ({12'd0, data_out_6} << 2);
assign CA00[19*20 +: 20] = ({12'd0, data_out_6} << 3);
assign CA00[20*20 +: 20] = ({12'd0, data_out_7} << 1);
assign CA00[21*20 +: 20] = ({12'd0, data_out_7} << 2);
assign CA00[22*20 +: 20] = ({12'd0, data_out_7} << 4);
assign CA00[23*20 +: 20] = ~({12'd0, data_out_8} << 1);
assign CA00[24*20 +: 20] = ~({12'd0, data_out_8} << 2);
assign CA00[25*20 +: 20] = ~({12'd0, data_out_8} << 3);
assign CA00[26*20 +: 20] = ~{12'd0, data_out_9};
assign CA00[27*20 +: 20] = ~({12'd0, data_out_9} << 3);
assign CA00[28*20 +: 20] = ~({12'd0, data_out_10} << 2);
assign CA00[29*20 +: 20] = ({12'd0, data_out_11} << 1);
assign CA00[30*20 +: 20] = ({12'd0, data_out_11} << 2);
assign CA00[31*20 +: 20] = ({12'd0, data_out_12} << 3);
assign CA00[32*20 +: 20] = ~{12'd0, data_out_13};
assign CA00[33*20 +: 20] = ~{12'd0, data_out_14};
assign CA00[34*20 +: 20] = ~({12'd0, data_out_14} << 1);
assign CA00[35*20 +: 20] = ~({12'd0, data_out_14} << 2);
assign CA00[36*20 +: 20] = ~({12'd0, data_out_14} << 4);
assign CA00[37*20 +: 20] = {12'd0, data_out_15};
assign CA00[38*20 +: 20] = ({12'd0, data_out_15} << 4);
assign CA00[39*20 +: 20] = {12'd0, data_out_16};
assign CA00[40*20 +: 20] = ~({12'd0, data_out_17} << 2);
assign CA00[41*20 +: 20] = ~({12'd0, data_out_17} << 4);
assign CA00[42*20 +: 20] = {12'd0, data_out_18};
assign CA00[43*20 +: 20] = ({12'd0, data_out_18} << 2);
assign CA00[44*20 +: 20] = ~({12'd0, data_out_19} << 2);
assign CA00[45*20 +: 20] = ~({12'd0, data_out_19} << 4);
assign CA00[46*20 +: 20] = ~{12'd0, data_out_20};
assign CA00[47*20 +: 20] = ~({12'd0, data_out_20} << 3);
assign CA00[48*20 +: 20] = ~{12'd0, data_out_21};
assign CA00[49*20 +: 20] = ~({12'd0, data_out_21} << 3);
assign CA00[50*20 +: 20] = ~({12'd0, data_out_21} << 4);
assign CA00[51*20 +: 20] = {12'd0, data_out_22};
assign CA00[52*20 +: 20] = ({12'd0, data_out_22} << 4);
assign CA00[53*20 +: 20] = ~({12'd0, data_out_23} << 1);
assign CA00[54*20 +: 20] = ~({12'd0, data_out_23} << 2);
assign CA00[55*20 +: 20] = ~({12'd0, data_out_24} << 2);
assign CA00[56*20 +: 20] = ~({12'd0, data_out_24} << 3);
assign CA00[57*20 +: 20] = $signed({exp_bias[0], 8'd0}) + 20'd34;
// 3:2 compression : 58 -> 39 -> 26 -> 18 -> 12 -> 8 -> 6 -> 4 -> 3 -> 2
genvar ca;
generate
  wire [39*20-1:0] CA01;
  for (ca = 0; ca < 19; ca = ca + 1) begin : CSA_CA_01
    wire [19:0] a = CA00[(3*ca+0)*20 +: 20];
    wire [19:0] b = CA00[(3*ca+1)*20 +: 20];
    wire [19:0] c = CA00[(3*ca+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CA01[(2*ca+0)*20 +: 20] = a ^ b ^ c;
    assign CA01[(2*ca+1)*20 +: 20] = cy << 1;
  end
  assign CA01[38*20 +: 20] = CA00[57*20 +: 20];
  wire [26*20-1:0] CA02;
  for (ca = 0; ca < 13; ca = ca + 1) begin : CSA_CA_02
    wire [19:0] a = CA01[(3*ca+0)*20 +: 20];
    wire [19:0] b = CA01[(3*ca+1)*20 +: 20];
    wire [19:0] c = CA01[(3*ca+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CA02[(2*ca+0)*20 +: 20] = a ^ b ^ c;
    assign CA02[(2*ca+1)*20 +: 20] = cy << 1;
  end
  wire [18*20-1:0] CA03;
  for (ca = 0; ca < 8; ca = ca + 1) begin : CSA_CA_03
    wire [19:0] a = CA02[(3*ca+0)*20 +: 20];
    wire [19:0] b = CA02[(3*ca+1)*20 +: 20];
    wire [19:0] c = CA02[(3*ca+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CA03[(2*ca+0)*20 +: 20] = a ^ b ^ c;
    assign CA03[(2*ca+1)*20 +: 20] = cy << 1;
  end
  assign CA03[16*20 +: 20] = CA02[24*20 +: 20];
  assign CA03[17*20 +: 20] = CA02[25*20 +: 20];
  wire [12*20-1:0] CA04;
  for (ca = 0; ca < 6; ca = ca + 1) begin : CSA_CA_04
    wire [19:0] a = CA03[(3*ca+0)*20 +: 20];
    wire [19:0] b = CA03[(3*ca+1)*20 +: 20];
    wire [19:0] c = CA03[(3*ca+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CA04[(2*ca+0)*20 +: 20] = a ^ b ^ c;
    assign CA04[(2*ca+1)*20 +: 20] = cy << 1;
  end
  wire [8*20-1:0] CA05;
  for (ca = 0; ca < 4; ca = ca + 1) begin : CSA_CA_05
    wire [19:0] a = CA04[(3*ca+0)*20 +: 20];
    wire [19:0] b = CA04[(3*ca+1)*20 +: 20];
    wire [19:0] c = CA04[(3*ca+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CA05[(2*ca+0)*20 +: 20] = a ^ b ^ c;
    assign CA05[(2*ca+1)*20 +: 20] = cy << 1;
  end
  wire [6*20-1:0] CA06;
  for (ca = 0; ca < 2; ca = ca + 1) begin : CSA_CA_06
    wire [19:0] a = CA05[(3*ca+0)*20 +: 20];
    wire [19:0] b = CA05[(3*ca+1)*20 +: 20];
    wire [19:0] c = CA05[(3*ca+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CA06[(2*ca+0)*20 +: 20] = a ^ b ^ c;
    assign CA06[(2*ca+1)*20 +: 20] = cy << 1;
  end
  assign CA06[4*20 +: 20] = CA05[6*20 +: 20];
  assign CA06[5*20 +: 20] = CA05[7*20 +: 20];
  wire [4*20-1:0] CA07;
  for (ca = 0; ca < 2; ca = ca + 1) begin : CSA_CA_07
    wire [19:0] a = CA06[(3*ca+0)*20 +: 20];
    wire [19:0] b = CA06[(3*ca+1)*20 +: 20];
    wire [19:0] c = CA06[(3*ca+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CA07[(2*ca+0)*20 +: 20] = a ^ b ^ c;
    assign CA07[(2*ca+1)*20 +: 20] = cy << 1;
  end
  wire [3*20-1:0] CA08;
  for (ca = 0; ca < 1; ca = ca + 1) begin : CSA_CA_08
    wire [19:0] a = CA07[(3*ca+0)*20 +: 20];
    wire [19:0] b = CA07[(3*ca+1)*20 +: 20];
    wire [19:0] c = CA07[(3*ca+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CA08[(2*ca+0)*20 +: 20] = a ^ b ^ c;
    assign CA08[(2*ca+1)*20 +: 20] = cy << 1;
  end
  assign CA08[2*20 +: 20] = CA07[3*20 +: 20];
  wire [2*20-1:0] CA09;
  for (ca = 0; ca < 1; ca = ca + 1) begin : CSA_CA_09
    wire [19:0] a = CA08[(3*ca+0)*20 +: 20];
    wire [19:0] b = CA08[(3*ca+1)*20 +: 20];
    wire [19:0] c = CA08[(3*ca+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CA09[(2*ca+0)*20 +: 20] = a ^ b ^ c;
    assign CA09[(2*ca+1)*20 +: 20] = cy << 1;
  end
endgenerate
wire [19:0] CAg [0:5];
wire [19:0] CAp [0:5];
assign CAg[0] = CA09[0*20 +: 20] & CA09[1*20 +: 20];
assign CAp[0] = CA09[0*20 +: 20] ^ CA09[1*20 +: 20];
genvar cal, cai;
generate
  for (cal = 1; cal <= 5; cal = cal + 1) begin : KSL_CA
    for (cai = 0; cai < 20; cai = cai + 1) begin : KSB
      if (cai >= (1 << (cal-1))) begin : UPD
        assign CAg[cal][cai] = CAg[cal-1][cai] | (CAp[cal-1][cai] & CAg[cal-1][cai-(1<<(cal-1))]);
        assign CAp[cal][cai] = CAp[cal-1][cai] & CAp[cal-1][cai-(1<<(cal-1))];
      end else begin : KEEP
        assign CAg[cal][cai] = CAg[cal-1][cai];
        assign CAp[cal][cai] = CAp[cal-1][cai];
      end
    end
  end
endgenerate
wire [19:0] CA_sum;
assign CA_sum[0] = CAp[0][0];
generate
  for (cai = 1; cai < 20; cai = cai + 1) begin : KSS_CA
    assign CA_sum[cai] = CAp[0][cai] ^ CAg[5][cai-1];
  end
endgenerate

localparam NR_CB = 73;
wire [NR_CB*20-1:0] CB00;
assign CB00[0*20 +: 20] = ~{12'd0, data_out_1};
assign CB00[1*20 +: 20] = ~({12'd0, data_out_1} << 1);
assign CB00[2*20 +: 20] = ~({12'd0, data_out_1} << 2);
assign CB00[3*20 +: 20] = ~({12'd0, data_out_1} << 3);
assign CB00[4*20 +: 20] = {12'd0, data_out_2};
assign CB00[5*20 +: 20] = ({12'd0, data_out_2} << 3);
assign CB00[6*20 +: 20] = ({12'd0, data_out_2} << 5);
assign CB00[7*20 +: 20] = ({12'd0, data_out_3} << 2);
assign CB00[8*20 +: 20] = ({12'd0, data_out_3} << 3);
assign CB00[9*20 +: 20] = ({12'd0, data_out_3} << 4);
assign CB00[10*20 +: 20] = ({12'd0, data_out_3} << 6);
assign CB00[11*20 +: 20] = {12'd0, data_out_4};
assign CB00[12*20 +: 20] = ({12'd0, data_out_4} << 1);
assign CB00[13*20 +: 20] = ({12'd0, data_out_4} << 2);
assign CB00[14*20 +: 20] = ({12'd0, data_out_4} << 3);
assign CB00[15*20 +: 20] = ({12'd0, data_out_4} << 6);
assign CB00[16*20 +: 20] = ~({12'd0, data_out_5} << 1);
assign CB00[17*20 +: 20] = ~({12'd0, data_out_5} << 3);
assign CB00[18*20 +: 20] = ~({12'd0, data_out_5} << 4);
assign CB00[19*20 +: 20] = ~({12'd0, data_out_6} << 5);
assign CB00[20*20 +: 20] = {12'd0, data_out_7};
assign CB00[21*20 +: 20] = ({12'd0, data_out_7} << 2);
assign CB00[22*20 +: 20] = ({12'd0, data_out_7} << 3);
assign CB00[23*20 +: 20] = ({12'd0, data_out_7} << 4);
assign CB00[24*20 +: 20] = ({12'd0, data_out_8} << 4);
assign CB00[25*20 +: 20] = ({12'd0, data_out_8} << 6);
assign CB00[26*20 +: 20] = {12'd0, data_out_9};
assign CB00[27*20 +: 20] = ({12'd0, data_out_9} << 1);
assign CB00[28*20 +: 20] = ({12'd0, data_out_9} << 2);
assign CB00[29*20 +: 20] = ({12'd0, data_out_9} << 6);
assign CB00[30*20 +: 20] = ~({12'd0, data_out_10} << 5);
assign CB00[31*20 +: 20] = {12'd0, data_out_11};
assign CB00[32*20 +: 20] = ({12'd0, data_out_11} << 2);
assign CB00[33*20 +: 20] = {12'd0, data_out_12};
assign CB00[34*20 +: 20] = ({12'd0, data_out_12} << 1);
assign CB00[35*20 +: 20] = ({12'd0, data_out_12} << 5);
assign CB00[36*20 +: 20] = ({12'd0, data_out_13} << 5);
assign CB00[37*20 +: 20] = ({12'd0, data_out_13} << 6);
assign CB00[38*20 +: 20] = {12'd0, data_out_14};
assign CB00[39*20 +: 20] = ({12'd0, data_out_14} << 5);
assign CB00[40*20 +: 20] = ~{12'd0, data_out_15};
assign CB00[41*20 +: 20] = ~({12'd0, data_out_15} << 1);
assign CB00[42*20 +: 20] = ({12'd0, data_out_16} << 1);
assign CB00[43*20 +: 20] = ({12'd0, data_out_16} << 3);
assign CB00[44*20 +: 20] = ({12'd0, data_out_16} << 4);
assign CB00[45*20 +: 20] = {12'd0, data_out_17};
assign CB00[46*20 +: 20] = ({12'd0, data_out_17} << 2);
assign CB00[47*20 +: 20] = ({12'd0, data_out_17} << 4);
assign CB00[48*20 +: 20] = ({12'd0, data_out_17} << 5);
assign CB00[49*20 +: 20] = ({12'd0, data_out_18} << 1);
assign CB00[50*20 +: 20] = ({12'd0, data_out_18} << 3);
assign CB00[51*20 +: 20] = ({12'd0, data_out_18} << 4);
assign CB00[52*20 +: 20] = ({12'd0, data_out_18} << 6);
assign CB00[53*20 +: 20] = ({12'd0, data_out_19} << 1);
assign CB00[54*20 +: 20] = ({12'd0, data_out_19} << 2);
assign CB00[55*20 +: 20] = {12'd0, data_out_20};
assign CB00[56*20 +: 20] = ({12'd0, data_out_20} << 1);
assign CB00[57*20 +: 20] = ({12'd0, data_out_20} << 2);
assign CB00[58*20 +: 20] = ({12'd0, data_out_20} << 4);
assign CB00[59*20 +: 20] = ({12'd0, data_out_21} << 2);
assign CB00[60*20 +: 20] = ({12'd0, data_out_21} << 3);
assign CB00[61*20 +: 20] = ({12'd0, data_out_21} << 4);
assign CB00[62*20 +: 20] = {12'd0, data_out_22};
assign CB00[63*20 +: 20] = ({12'd0, data_out_22} << 2);
assign CB00[64*20 +: 20] = ({12'd0, data_out_22} << 3);
assign CB00[65*20 +: 20] = ({12'd0, data_out_22} << 5);
assign CB00[66*20 +: 20] = ({12'd0, data_out_23} << 2);
assign CB00[67*20 +: 20] = ({12'd0, data_out_23} << 4);
assign CB00[68*20 +: 20] = ({12'd0, data_out_23} << 5);
assign CB00[69*20 +: 20] = ({12'd0, data_out_24} << 2);
assign CB00[70*20 +: 20] = ({12'd0, data_out_24} << 3);
assign CB00[71*20 +: 20] = ({12'd0, data_out_24} << 5);
assign CB00[72*20 +: 20] = $signed({exp_bias[1], 8'd0}) + 20'd11;
// 3:2 compression : 73 -> 49 -> 33 -> 22 -> 15 -> 10 -> 7 -> 5 -> 4 -> 3 -> 2
genvar cb;
generate
  wire [49*20-1:0] CB01;
  for (cb = 0; cb < 24; cb = cb + 1) begin : CSA_CB_01
    wire [19:0] a = CB00[(3*cb+0)*20 +: 20];
    wire [19:0] b = CB00[(3*cb+1)*20 +: 20];
    wire [19:0] c = CB00[(3*cb+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CB01[(2*cb+0)*20 +: 20] = a ^ b ^ c;
    assign CB01[(2*cb+1)*20 +: 20] = cy << 1;
  end
  assign CB01[48*20 +: 20] = CB00[72*20 +: 20];
  wire [33*20-1:0] CB02;
  for (cb = 0; cb < 16; cb = cb + 1) begin : CSA_CB_02
    wire [19:0] a = CB01[(3*cb+0)*20 +: 20];
    wire [19:0] b = CB01[(3*cb+1)*20 +: 20];
    wire [19:0] c = CB01[(3*cb+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CB02[(2*cb+0)*20 +: 20] = a ^ b ^ c;
    assign CB02[(2*cb+1)*20 +: 20] = cy << 1;
  end
  assign CB02[32*20 +: 20] = CB01[48*20 +: 20];
  wire [22*20-1:0] CB03;
  for (cb = 0; cb < 11; cb = cb + 1) begin : CSA_CB_03
    wire [19:0] a = CB02[(3*cb+0)*20 +: 20];
    wire [19:0] b = CB02[(3*cb+1)*20 +: 20];
    wire [19:0] c = CB02[(3*cb+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CB03[(2*cb+0)*20 +: 20] = a ^ b ^ c;
    assign CB03[(2*cb+1)*20 +: 20] = cy << 1;
  end
  wire [15*20-1:0] CB04;
  for (cb = 0; cb < 7; cb = cb + 1) begin : CSA_CB_04
    wire [19:0] a = CB03[(3*cb+0)*20 +: 20];
    wire [19:0] b = CB03[(3*cb+1)*20 +: 20];
    wire [19:0] c = CB03[(3*cb+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CB04[(2*cb+0)*20 +: 20] = a ^ b ^ c;
    assign CB04[(2*cb+1)*20 +: 20] = cy << 1;
  end
  assign CB04[14*20 +: 20] = CB03[21*20 +: 20];
  wire [10*20-1:0] CB05;
  for (cb = 0; cb < 5; cb = cb + 1) begin : CSA_CB_05
    wire [19:0] a = CB04[(3*cb+0)*20 +: 20];
    wire [19:0] b = CB04[(3*cb+1)*20 +: 20];
    wire [19:0] c = CB04[(3*cb+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CB05[(2*cb+0)*20 +: 20] = a ^ b ^ c;
    assign CB05[(2*cb+1)*20 +: 20] = cy << 1;
  end
  wire [7*20-1:0] CB06;
  for (cb = 0; cb < 3; cb = cb + 1) begin : CSA_CB_06
    wire [19:0] a = CB05[(3*cb+0)*20 +: 20];
    wire [19:0] b = CB05[(3*cb+1)*20 +: 20];
    wire [19:0] c = CB05[(3*cb+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CB06[(2*cb+0)*20 +: 20] = a ^ b ^ c;
    assign CB06[(2*cb+1)*20 +: 20] = cy << 1;
  end
  assign CB06[6*20 +: 20] = CB05[9*20 +: 20];
  wire [5*20-1:0] CB07;
  for (cb = 0; cb < 2; cb = cb + 1) begin : CSA_CB_07
    wire [19:0] a = CB06[(3*cb+0)*20 +: 20];
    wire [19:0] b = CB06[(3*cb+1)*20 +: 20];
    wire [19:0] c = CB06[(3*cb+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CB07[(2*cb+0)*20 +: 20] = a ^ b ^ c;
    assign CB07[(2*cb+1)*20 +: 20] = cy << 1;
  end
  assign CB07[4*20 +: 20] = CB06[6*20 +: 20];
  wire [4*20-1:0] CB08;
  for (cb = 0; cb < 1; cb = cb + 1) begin : CSA_CB_08
    wire [19:0] a = CB07[(3*cb+0)*20 +: 20];
    wire [19:0] b = CB07[(3*cb+1)*20 +: 20];
    wire [19:0] c = CB07[(3*cb+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CB08[(2*cb+0)*20 +: 20] = a ^ b ^ c;
    assign CB08[(2*cb+1)*20 +: 20] = cy << 1;
  end
  assign CB08[2*20 +: 20] = CB07[3*20 +: 20];
  assign CB08[3*20 +: 20] = CB07[4*20 +: 20];
  wire [3*20-1:0] CB09;
  for (cb = 0; cb < 1; cb = cb + 1) begin : CSA_CB_09
    wire [19:0] a = CB08[(3*cb+0)*20 +: 20];
    wire [19:0] b = CB08[(3*cb+1)*20 +: 20];
    wire [19:0] c = CB08[(3*cb+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CB09[(2*cb+0)*20 +: 20] = a ^ b ^ c;
    assign CB09[(2*cb+1)*20 +: 20] = cy << 1;
  end
  assign CB09[2*20 +: 20] = CB08[3*20 +: 20];
  wire [2*20-1:0] CB10;
  for (cb = 0; cb < 1; cb = cb + 1) begin : CSA_CB_10
    wire [19:0] a = CB09[(3*cb+0)*20 +: 20];
    wire [19:0] b = CB09[(3*cb+1)*20 +: 20];
    wire [19:0] c = CB09[(3*cb+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CB10[(2*cb+0)*20 +: 20] = a ^ b ^ c;
    assign CB10[(2*cb+1)*20 +: 20] = cy << 1;
  end
endgenerate
wire [19:0] CBg [0:5];
wire [19:0] CBp [0:5];
assign CBg[0] = CB10[0*20 +: 20] & CB10[1*20 +: 20];
assign CBp[0] = CB10[0*20 +: 20] ^ CB10[1*20 +: 20];
genvar cbl, cbi;
generate
  for (cbl = 1; cbl <= 5; cbl = cbl + 1) begin : KSL_CB
    for (cbi = 0; cbi < 20; cbi = cbi + 1) begin : KSB
      if (cbi >= (1 << (cbl-1))) begin : UPD
        assign CBg[cbl][cbi] = CBg[cbl-1][cbi] | (CBp[cbl-1][cbi] & CBg[cbl-1][cbi-(1<<(cbl-1))]);
        assign CBp[cbl][cbi] = CBp[cbl-1][cbi] & CBp[cbl-1][cbi-(1<<(cbl-1))];
      end else begin : KEEP
        assign CBg[cbl][cbi] = CBg[cbl-1][cbi];
        assign CBp[cbl][cbi] = CBp[cbl-1][cbi];
      end
    end
  end
endgenerate
wire [19:0] CB_sum;
assign CB_sum[0] = CBp[0][0];
generate
  for (cbi = 1; cbi < 20; cbi = cbi + 1) begin : KSS_CB
    assign CB_sum[cbi] = CBp[0][cbi] ^ CBg[5][cbi-1];
  end
endgenerate

localparam NR_CC = 74;
wire [NR_CC*20-1:0] CC00;
assign CC00[0*20 +: 20] = {12'd0, data_out_0};
assign CC00[1*20 +: 20] = ({12'd0, data_out_0} << 2);
assign CC00[2*20 +: 20] = ({12'd0, data_out_0} << 3);
assign CC00[3*20 +: 20] = ({12'd0, data_out_0} << 4);
assign CC00[4*20 +: 20] = ({12'd0, data_out_1} << 3);
assign CC00[5*20 +: 20] = ~({12'd0, data_out_2} << 3);
assign CC00[6*20 +: 20] = ~({12'd0, data_out_2} << 4);
assign CC00[7*20 +: 20] = ~({12'd0, data_out_3} << 3);
assign CC00[8*20 +: 20] = ~({12'd0, data_out_3} << 4);
assign CC00[9*20 +: 20] = ~({12'd0, data_out_4} << 3);
assign CC00[10*20 +: 20] = ~({12'd0, data_out_4} << 4);
assign CC00[11*20 +: 20] = ({12'd0, data_out_5} << 1);
assign CC00[12*20 +: 20] = ({12'd0, data_out_5} << 3);
assign CC00[13*20 +: 20] = ({12'd0, data_out_5} << 5);
assign CC00[14*20 +: 20] = {12'd0, data_out_6};
assign CC00[15*20 +: 20] = ({12'd0, data_out_6} << 1);
assign CC00[16*20 +: 20] = ({12'd0, data_out_6} << 3);
assign CC00[17*20 +: 20] = ({12'd0, data_out_6} << 5);
assign CC00[18*20 +: 20] = ~{12'd0, data_out_7};
assign CC00[19*20 +: 20] = ~({12'd0, data_out_7} << 2);
assign CC00[20*20 +: 20] = ~({12'd0, data_out_7} << 3);
assign CC00[21*20 +: 20] = ~({12'd0, data_out_7} << 4);
assign CC00[22*20 +: 20] = ~({12'd0, data_out_8} << 2);
assign CC00[23*20 +: 20] = ~({12'd0, data_out_8} << 3);
assign CC00[24*20 +: 20] = ~({12'd0, data_out_8} << 6);
assign CC00[25*20 +: 20] = ~({12'd0, data_out_9} << 2);
assign CC00[26*20 +: 20] = ~({12'd0, data_out_9} << 3);
assign CC00[27*20 +: 20] = ~({12'd0, data_out_9} << 6);
assign CC00[28*20 +: 20] = ({12'd0, data_out_10} << 1);
assign CC00[29*20 +: 20] = ({12'd0, data_out_10} << 2);
assign CC00[30*20 +: 20] = ({12'd0, data_out_10} << 4);
assign CC00[31*20 +: 20] = ({12'd0, data_out_11} << 2);
assign CC00[32*20 +: 20] = ({12'd0, data_out_11} << 3);
assign CC00[33*20 +: 20] = ({12'd0, data_out_11} << 4);
assign CC00[34*20 +: 20] = ~{12'd0, data_out_12};
assign CC00[35*20 +: 20] = ~({12'd0, data_out_12} << 1);
assign CC00[36*20 +: 20] = ~({12'd0, data_out_12} << 2);
assign CC00[37*20 +: 20] = ({12'd0, data_out_13} << 1);
assign CC00[38*20 +: 20] = ({12'd0, data_out_13} << 3);
assign CC00[39*20 +: 20] = ~({12'd0, data_out_14} << 2);
assign CC00[40*20 +: 20] = ({12'd0, data_out_15} << 3);
assign CC00[41*20 +: 20] = ({12'd0, data_out_15} << 4);
assign CC00[42*20 +: 20] = {12'd0, data_out_16};
assign CC00[43*20 +: 20] = ({12'd0, data_out_16} << 2);
assign CC00[44*20 +: 20] = ({12'd0, data_out_16} << 3);
assign CC00[45*20 +: 20] = ({12'd0, data_out_16} << 4);
assign CC00[46*20 +: 20] = ({12'd0, data_out_17} << 1);
assign CC00[47*20 +: 20] = ({12'd0, data_out_17} << 3);
assign CC00[48*20 +: 20] = ({12'd0, data_out_17} << 4);
assign CC00[49*20 +: 20] = ({12'd0, data_out_17} << 5);
assign CC00[50*20 +: 20] = {12'd0, data_out_18};
assign CC00[51*20 +: 20] = ({12'd0, data_out_18} << 2);
assign CC00[52*20 +: 20] = ({12'd0, data_out_18} << 3);
assign CC00[53*20 +: 20] = ({12'd0, data_out_18} << 6);
assign CC00[54*20 +: 20] = ({12'd0, data_out_19} << 1);
assign CC00[55*20 +: 20] = ({12'd0, data_out_19} << 3);
assign CC00[56*20 +: 20] = ({12'd0, data_out_19} << 4);
assign CC00[57*20 +: 20] = ({12'd0, data_out_19} << 5);
assign CC00[58*20 +: 20] = ({12'd0, data_out_20} << 1);
assign CC00[59*20 +: 20] = {12'd0, data_out_21};
assign CC00[60*20 +: 20] = ({12'd0, data_out_21} << 3);
assign CC00[61*20 +: 20] = ({12'd0, data_out_21} << 4);
assign CC00[62*20 +: 20] = ({12'd0, data_out_21} << 5);
assign CC00[63*20 +: 20] = {12'd0, data_out_22};
assign CC00[64*20 +: 20] = ({12'd0, data_out_22} << 2);
assign CC00[65*20 +: 20] = ({12'd0, data_out_22} << 6);
assign CC00[66*20 +: 20] = {12'd0, data_out_23};
assign CC00[67*20 +: 20] = ({12'd0, data_out_23} << 1);
assign CC00[68*20 +: 20] = ({12'd0, data_out_23} << 4);
assign CC00[69*20 +: 20] = ({12'd0, data_out_23} << 6);
assign CC00[70*20 +: 20] = ({12'd0, data_out_24} << 1);
assign CC00[71*20 +: 20] = ({12'd0, data_out_24} << 3);
assign CC00[72*20 +: 20] = ({12'd0, data_out_24} << 5);
assign CC00[73*20 +: 20] = $signed({exp_bias[2], 8'd0}) + 20'd20;
// 3:2 compression : 74 -> 50 -> 34 -> 23 -> 16 -> 11 -> 8 -> 6 -> 4 -> 3 -> 2
genvar cc;
generate
  wire [50*20-1:0] CC01;
  for (cc = 0; cc < 24; cc = cc + 1) begin : CSA_CC_01
    wire [19:0] a = CC00[(3*cc+0)*20 +: 20];
    wire [19:0] b = CC00[(3*cc+1)*20 +: 20];
    wire [19:0] c = CC00[(3*cc+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CC01[(2*cc+0)*20 +: 20] = a ^ b ^ c;
    assign CC01[(2*cc+1)*20 +: 20] = cy << 1;
  end
  assign CC01[48*20 +: 20] = CC00[72*20 +: 20];
  assign CC01[49*20 +: 20] = CC00[73*20 +: 20];
  wire [34*20-1:0] CC02;
  for (cc = 0; cc < 16; cc = cc + 1) begin : CSA_CC_02
    wire [19:0] a = CC01[(3*cc+0)*20 +: 20];
    wire [19:0] b = CC01[(3*cc+1)*20 +: 20];
    wire [19:0] c = CC01[(3*cc+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CC02[(2*cc+0)*20 +: 20] = a ^ b ^ c;
    assign CC02[(2*cc+1)*20 +: 20] = cy << 1;
  end
  assign CC02[32*20 +: 20] = CC01[48*20 +: 20];
  assign CC02[33*20 +: 20] = CC01[49*20 +: 20];
  wire [23*20-1:0] CC03;
  for (cc = 0; cc < 11; cc = cc + 1) begin : CSA_CC_03
    wire [19:0] a = CC02[(3*cc+0)*20 +: 20];
    wire [19:0] b = CC02[(3*cc+1)*20 +: 20];
    wire [19:0] c = CC02[(3*cc+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CC03[(2*cc+0)*20 +: 20] = a ^ b ^ c;
    assign CC03[(2*cc+1)*20 +: 20] = cy << 1;
  end
  assign CC03[22*20 +: 20] = CC02[33*20 +: 20];
  wire [16*20-1:0] CC04;
  for (cc = 0; cc < 7; cc = cc + 1) begin : CSA_CC_04
    wire [19:0] a = CC03[(3*cc+0)*20 +: 20];
    wire [19:0] b = CC03[(3*cc+1)*20 +: 20];
    wire [19:0] c = CC03[(3*cc+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CC04[(2*cc+0)*20 +: 20] = a ^ b ^ c;
    assign CC04[(2*cc+1)*20 +: 20] = cy << 1;
  end
  assign CC04[14*20 +: 20] = CC03[21*20 +: 20];
  assign CC04[15*20 +: 20] = CC03[22*20 +: 20];
  wire [11*20-1:0] CC05;
  for (cc = 0; cc < 5; cc = cc + 1) begin : CSA_CC_05
    wire [19:0] a = CC04[(3*cc+0)*20 +: 20];
    wire [19:0] b = CC04[(3*cc+1)*20 +: 20];
    wire [19:0] c = CC04[(3*cc+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CC05[(2*cc+0)*20 +: 20] = a ^ b ^ c;
    assign CC05[(2*cc+1)*20 +: 20] = cy << 1;
  end
  assign CC05[10*20 +: 20] = CC04[15*20 +: 20];
  wire [8*20-1:0] CC06;
  for (cc = 0; cc < 3; cc = cc + 1) begin : CSA_CC_06
    wire [19:0] a = CC05[(3*cc+0)*20 +: 20];
    wire [19:0] b = CC05[(3*cc+1)*20 +: 20];
    wire [19:0] c = CC05[(3*cc+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CC06[(2*cc+0)*20 +: 20] = a ^ b ^ c;
    assign CC06[(2*cc+1)*20 +: 20] = cy << 1;
  end
  assign CC06[6*20 +: 20] = CC05[9*20 +: 20];
  assign CC06[7*20 +: 20] = CC05[10*20 +: 20];
  wire [6*20-1:0] CC07;
  for (cc = 0; cc < 2; cc = cc + 1) begin : CSA_CC_07
    wire [19:0] a = CC06[(3*cc+0)*20 +: 20];
    wire [19:0] b = CC06[(3*cc+1)*20 +: 20];
    wire [19:0] c = CC06[(3*cc+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CC07[(2*cc+0)*20 +: 20] = a ^ b ^ c;
    assign CC07[(2*cc+1)*20 +: 20] = cy << 1;
  end
  assign CC07[4*20 +: 20] = CC06[6*20 +: 20];
  assign CC07[5*20 +: 20] = CC06[7*20 +: 20];
  wire [4*20-1:0] CC08;
  for (cc = 0; cc < 2; cc = cc + 1) begin : CSA_CC_08
    wire [19:0] a = CC07[(3*cc+0)*20 +: 20];
    wire [19:0] b = CC07[(3*cc+1)*20 +: 20];
    wire [19:0] c = CC07[(3*cc+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CC08[(2*cc+0)*20 +: 20] = a ^ b ^ c;
    assign CC08[(2*cc+1)*20 +: 20] = cy << 1;
  end
  wire [3*20-1:0] CC09;
  for (cc = 0; cc < 1; cc = cc + 1) begin : CSA_CC_09
    wire [19:0] a = CC08[(3*cc+0)*20 +: 20];
    wire [19:0] b = CC08[(3*cc+1)*20 +: 20];
    wire [19:0] c = CC08[(3*cc+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CC09[(2*cc+0)*20 +: 20] = a ^ b ^ c;
    assign CC09[(2*cc+1)*20 +: 20] = cy << 1;
  end
  assign CC09[2*20 +: 20] = CC08[3*20 +: 20];
  wire [2*20-1:0] CC10;
  for (cc = 0; cc < 1; cc = cc + 1) begin : CSA_CC_10
    wire [19:0] a = CC09[(3*cc+0)*20 +: 20];
    wire [19:0] b = CC09[(3*cc+1)*20 +: 20];
    wire [19:0] c = CC09[(3*cc+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign CC10[(2*cc+0)*20 +: 20] = a ^ b ^ c;
    assign CC10[(2*cc+1)*20 +: 20] = cy << 1;
  end
endgenerate
wire [19:0] CCg [0:5];
wire [19:0] CCp [0:5];
assign CCg[0] = CC10[0*20 +: 20] & CC10[1*20 +: 20];
assign CCp[0] = CC10[0*20 +: 20] ^ CC10[1*20 +: 20];
genvar ccl, cci;
generate
  for (ccl = 1; ccl <= 5; ccl = ccl + 1) begin : KSL_CC
    for (cci = 0; cci < 20; cci = cci + 1) begin : KSB
      if (cci >= (1 << (ccl-1))) begin : UPD
        assign CCg[ccl][cci] = CCg[ccl-1][cci] | (CCp[ccl-1][cci] & CCg[ccl-1][cci-(1<<(ccl-1))]);
        assign CCp[ccl][cci] = CCp[ccl-1][cci] & CCp[ccl-1][cci-(1<<(ccl-1))];
      end else begin : KEEP
        assign CCg[ccl][cci] = CCg[ccl-1][cci];
        assign CCp[ccl][cci] = CCp[ccl-1][cci];
      end
    end
  end
endgenerate
wire [19:0] CC_sum;
assign CC_sum[0] = CCp[0][0];
generate
  for (cci = 1; cci < 20; cci = cci + 1) begin : KSS_CC
    assign CC_sum[cci] = CCp[0][cci] ^ CCg[5][cci-1];
  end
endgenerate

 always @(posedge clk) begin
   if(~rst_n) begin
     valid_out_calc <= 1'b0;
   end else begin
     valid_out_calc <= valid_out_buf;
   end

   if(valid_out_buf) begin
     conv_out_1 <= CA_sum[19:8];
     conv_out_2 <= CB_sum[19:8];
     conv_out_3 <= CC_sum[19:8];
   end
 end

endmodule

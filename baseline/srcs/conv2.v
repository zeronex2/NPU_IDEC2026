/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : conv2_layer.v
 *  Written by : Kang, Bo Young
 *  Written on : Oct 11, 2021
 *  Version    : 21.2
 *  Design     : 2nd Convolution Layer for CNN MNIST dataset
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: conv2_layer
 *------------------------------------------------------------------*/
 
 module conv2_layer (
   input clk,
   input rst_n,
   input valid_in,
   input [11:0] max_value_1, max_value_2, max_value_3,
   output [11:0] conv2_out_1, conv2_out_2, conv2_out_3,
   output  valid_out_conv2
 );

 localparam CHANNEL_LEN = 3;

 // Trained bias, hard-wired as a constant.
 // Regenerate with gen_weights.py after re-quantizing the model.
 localparam [0:23] B_2 = 24'h15FFF6;
 ///////////////////////////////////////////
  /*wire [11:0] out_data1_0, out_data1_1, out_data1_2, out_data1_3, out_data1_4,
			out_data1_5, out_data1_6, out_data1_7, out_data1_8, out_data1_9,
			out_data1_10, out_data1_11, out_data1_12, out_data1_13, out_data1_14,
			out_data1_15, out_data1_16, out_data1_17, out_data1_18, out_data1_19,
			out_data1_20, out_data1_21, out_data1_22, out_data1_23, out_data1_24,
			
			out_data2_0, out_data2_1, out_data2_2, out_data2_3, out_data2_4,
			out_data2_5, out_data2_6, out_data2_7, out_data2_8, out_data2_9,
			out_data2_10, out_data2_11, out_data2_12, out_data2_13, out_data2_14,
			out_data2_15, out_data2_16, out_data2_17, out_data2_18, out_data2_19,
			out_data2_20, out_data2_21, out_data2_22, out_data2_23, out_data2_24,
			
			out_data3_0, out_data3_1, out_data3_2, out_data3_3, out_data3_4,
			out_data3_5, out_data3_6, out_data3_7, out_data3_8, out_data3_9,
			out_data3_10, out_data3_11, out_data3_12, out_data3_13, out_data3_14,
			out_data3_15, out_data3_16, out_data3_17, out_data3_18, out_data3_19,
			out_data3_20, out_data3_21, out_data3_22, out_data3_23, out_data3_24;*/
      /////////////////////////////////
 // Channel 1
 wire [11:0] data_out1_0, data_out1_1, data_out1_2, data_out1_3, data_out1_4,
  data_out1_5, data_out1_6, data_out1_7, data_out1_8, data_out1_9,
  data_out1_10, data_out1_11, data_out1_12, data_out1_13, data_out1_14,
  data_out1_15, data_out1_16, data_out1_17, data_out1_18, data_out1_19,
  data_out1_20, data_out1_21, data_out1_22, data_out1_23, data_out1_24;
 wire valid_out1_buf;

 // Channel 2
 wire [11:0] data_out2_0, data_out2_1, data_out2_2, data_out2_3, data_out2_4,
  data_out2_5, data_out2_6, data_out2_7, data_out2_8, data_out2_9,
  data_out2_10, data_out2_11, data_out2_12, data_out2_13, data_out2_14,
  data_out2_15, data_out2_16, data_out2_17, data_out2_18, data_out2_19,
  data_out2_20, data_out2_21, data_out2_22, data_out2_23, data_out2_24;
 wire valid_out2_buf;

 // Channel 3 
 wire [11:0] data_out3_0, data_out3_1, data_out3_2, data_out3_3, data_out3_4,
  data_out3_5, data_out3_6, data_out3_7, data_out3_8, data_out3_9,
  data_out3_10, data_out3_11, data_out3_12, data_out3_13, data_out3_14,
  data_out3_15, data_out3_16, data_out3_17, data_out3_18, data_out3_19,
  data_out3_20, data_out3_21, data_out3_22, data_out3_23, data_out3_24;
 wire valid_out3_buf;

 wire signed [13:0] conv_out_1, conv_out_2, conv_out_3;
 wire valid_out_buf, valid_out_calc_1, valid_out_calc_2, valid_out_calc_3;
 assign valid_out_buf = valid_out1_buf & valid_out2_buf & valid_out3_buf;
 assign valid_out_conv2 = valid_out_calc_1 & valid_out_calc_2 & valid_out_calc_3;

 wire signed [7:0] bias [0:CHANNEL_LEN - 1];
 wire signed [11:0] exp_bias [0:CHANNEL_LEN - 1];




conv2_buf #(.WIDTH(12), .HEIGHT(12), .DATA_BITS(12)) conv2_buf_1(
   .clk(clk),
   .rst_n(rst_n),
   .valid_in(valid_in),
   .data_in(max_value_1),
   .data_out_0(data_out1_0),
   .data_out_1(data_out1_1),
   .data_out_2(data_out1_2),
   .data_out_3(data_out1_3),
   .data_out_4(data_out1_4),
   .data_out_5(data_out1_5),
   .data_out_6(data_out1_6),
   .data_out_7(data_out1_7),
   .data_out_8(data_out1_8),
   .data_out_9(data_out1_9),
   .data_out_10(data_out1_10),
   .data_out_11(data_out1_11),
   .data_out_12(data_out1_12),
   .data_out_13(data_out1_13),
   .data_out_14(data_out1_14),
   .data_out_15(data_out1_15),
   .data_out_16(data_out1_16),
   .data_out_17(data_out1_17),
   .data_out_18(data_out1_18),
   .data_out_19(data_out1_19),
   .data_out_20(data_out1_20),
   .data_out_21(data_out1_21),
   .data_out_22(data_out1_22),
   .data_out_23(data_out1_23),
   .data_out_24(data_out1_24),
   .valid_out_buf(valid_out1_buf)
 );

 conv2_buf #(.WIDTH(12), .HEIGHT(12), .DATA_BITS  (12)) conv2_buf_2(
   .clk(clk),
   .rst_n(rst_n),
   .valid_in(valid_in),
   .data_in(max_value_2),
   .data_out_0(data_out2_0),
   .data_out_1(data_out2_1),
   .data_out_2(data_out2_2),
   .data_out_3(data_out2_3),
   .data_out_4(data_out2_4),
   .data_out_5(data_out2_5),
   .data_out_6(data_out2_6),
   .data_out_7(data_out2_7),
   .data_out_8(data_out2_8),
   .data_out_9(data_out2_9),
   .data_out_10(data_out2_10),
   .data_out_11(data_out2_11),
   .data_out_12(data_out2_12),
   .data_out_13(data_out2_13),
   .data_out_14(data_out2_14),
   .data_out_15(data_out2_15),
   .data_out_16(data_out2_16),
   .data_out_17(data_out2_17),
   .data_out_18(data_out2_18),
   .data_out_19(data_out2_19),
   .data_out_20(data_out2_20),
   .data_out_21(data_out2_21),
   .data_out_22(data_out2_22),
   .data_out_23(data_out2_23),
   .data_out_24(data_out2_24),
   .valid_out_buf(valid_out2_buf)
 );

 conv2_buf #(.WIDTH(12), .HEIGHT(12), .DATA_BITS(12)) conv2_buf_3(
   .clk(clk),
   .rst_n(rst_n),
   .valid_in(valid_in),
   .data_in(max_value_3),
   .data_out_0(data_out3_0),
   .data_out_1(data_out3_1),
   .data_out_2(data_out3_2),
   .data_out_3(data_out3_3),
   .data_out_4(data_out3_4),
   .data_out_5(data_out3_5),
   .data_out_6(data_out3_6),
   .data_out_7(data_out3_7),
   .data_out_8(data_out3_8),
   .data_out_9(data_out3_9),
   .data_out_10(data_out3_10),
   .data_out_11(data_out3_11),
   .data_out_12(data_out3_12),
   .data_out_13(data_out3_13),
   .data_out_14(data_out3_14),
   .data_out_15(data_out3_15),
   .data_out_16(data_out3_16),
   .data_out_17(data_out3_17),
   .data_out_18(data_out3_18),
   .data_out_19(data_out3_19),
   .data_out_20(data_out3_20),
   .data_out_21(data_out3_21),
   .data_out_22(data_out3_22),
   .data_out_23(data_out3_23),
   .data_out_24(data_out3_24),
   .valid_out_buf(valid_out3_buf)
 );

conv2_calc_1 conv2_calc_1(
   .clk(clk),
   .rst_n(rst_n),
   .valid_out_buf(valid_out_buf),
   .data_out1_0(data_out1_0),
   .data_out1_1(data_out1_1),
   .data_out1_2(data_out1_2),
   .data_out1_3(data_out1_3),
   .data_out1_4(data_out1_4),
   .data_out1_5(data_out1_5),
   .data_out1_6(data_out1_6),
   .data_out1_7(data_out1_7),
   .data_out1_8(data_out1_8),
   .data_out1_9(data_out1_9),
   .data_out1_10(data_out1_10),
   .data_out1_11(data_out1_11),
   .data_out1_12(data_out1_12),
   .data_out1_13(data_out1_13),
   .data_out1_14(data_out1_14),
   .data_out1_15(data_out1_15),
   .data_out1_16(data_out1_16),
   .data_out1_17(data_out1_17),
   .data_out1_18(data_out1_18),
   .data_out1_19(data_out1_19),
   .data_out1_20(data_out1_20),
   .data_out1_21(data_out1_21),
   .data_out1_22(data_out1_22),
   .data_out1_23(data_out1_23),
   .data_out1_24(data_out1_24),
   .data_out2_0(data_out2_0),
   .data_out2_1(data_out2_1),
   .data_out2_2(data_out2_2),
   .data_out2_3(data_out2_3),
   .data_out2_4(data_out2_4),
   .data_out2_5(data_out2_5),
   .data_out2_6(data_out2_6),
   .data_out2_7(data_out2_7),
   .data_out2_8(data_out2_8),
   .data_out2_9(data_out2_9),
   .data_out2_10(data_out2_10),
   .data_out2_11(data_out2_11),
   .data_out2_12(data_out2_12),
   .data_out2_13(data_out2_13),
   .data_out2_14(data_out2_14),
   .data_out2_15(data_out2_15),
   .data_out2_16(data_out2_16),
   .data_out2_17(data_out2_17),
   .data_out2_18(data_out2_18),
   .data_out2_19(data_out2_19),
   .data_out2_20(data_out2_20),
   .data_out2_21(data_out2_21),
   .data_out2_22(data_out2_22),
   .data_out2_23(data_out2_23),
   .data_out2_24(data_out2_24),
   .data_out3_0(data_out3_0),
   .data_out3_1(data_out3_1),
   .data_out3_2(data_out3_2),
   .data_out3_3(data_out3_3),
   .data_out3_4(data_out3_4),
   .data_out3_5(data_out3_5),
   .data_out3_6(data_out3_6),
   .data_out3_7(data_out3_7),
   .data_out3_8(data_out3_8),
   .data_out3_9(data_out3_9),
   .data_out3_10(data_out3_10),
   .data_out3_11(data_out3_11),
   .data_out3_12(data_out3_12),
   .data_out3_13(data_out3_13),
   .data_out3_14(data_out3_14),
   .data_out3_15(data_out3_15),
   .data_out3_16(data_out3_16),
   .data_out3_17(data_out3_17),
   .data_out3_18(data_out3_18),
   .data_out3_19(data_out3_19),
   .data_out3_20(data_out3_20),
   .data_out3_21(data_out3_21),
   .data_out3_22(data_out3_22),
   .data_out3_23(data_out3_23),
   .data_out3_24(data_out3_24),
   .conv_out_calc(conv_out_1),
   .valid_out_calc(valid_out_calc_1)
);

conv2_calc_2 conv2_calc_2(
   .clk(clk),
   .rst_n(rst_n),
   .valid_out_buf(valid_out_buf),
   .data_out1_0(data_out1_0),
   .data_out1_1(data_out1_1),
   .data_out1_2(data_out1_2),
   .data_out1_3(data_out1_3),
   .data_out1_4(data_out1_4),
   .data_out1_5(data_out1_5),
   .data_out1_6(data_out1_6),
   .data_out1_7(data_out1_7),
   .data_out1_8(data_out1_8),
   .data_out1_9(data_out1_9),
   .data_out1_10(data_out1_10),
   .data_out1_11(data_out1_11),
   .data_out1_12(data_out1_12),
   .data_out1_13(data_out1_13),
   .data_out1_14(data_out1_14),
   .data_out1_15(data_out1_15),
   .data_out1_16(data_out1_16),
   .data_out1_17(data_out1_17),
   .data_out1_18(data_out1_18),
   .data_out1_19(data_out1_19),
   .data_out1_20(data_out1_20),
   .data_out1_21(data_out1_21),
   .data_out1_22(data_out1_22),
   .data_out1_23(data_out1_23),
   .data_out1_24(data_out1_24),
   .data_out2_0(data_out2_0),
   .data_out2_1(data_out2_1),
   .data_out2_2(data_out2_2),
   .data_out2_3(data_out2_3),
   .data_out2_4(data_out2_4),
   .data_out2_5(data_out2_5),
   .data_out2_6(data_out2_6),
   .data_out2_7(data_out2_7),
   .data_out2_8(data_out2_8),
   .data_out2_9(data_out2_9),
   .data_out2_10(data_out2_10),
   .data_out2_11(data_out2_11),
   .data_out2_12(data_out2_12),
   .data_out2_13(data_out2_13),
   .data_out2_14(data_out2_14),
   .data_out2_15(data_out2_15),
   .data_out2_16(data_out2_16),
   .data_out2_17(data_out2_17),
   .data_out2_18(data_out2_18),
   .data_out2_19(data_out2_19),
   .data_out2_20(data_out2_20),
   .data_out2_21(data_out2_21),
   .data_out2_22(data_out2_22),
   .data_out2_23(data_out2_23),
   .data_out2_24(data_out2_24),
   .data_out3_0(data_out3_0),
   .data_out3_1(data_out3_1),
   .data_out3_2(data_out3_2),
   .data_out3_3(data_out3_3),
   .data_out3_4(data_out3_4),
   .data_out3_5(data_out3_5),
   .data_out3_6(data_out3_6),
   .data_out3_7(data_out3_7),
   .data_out3_8(data_out3_8),
   .data_out3_9(data_out3_9),
   .data_out3_10(data_out3_10),
   .data_out3_11(data_out3_11),
   .data_out3_12(data_out3_12),
   .data_out3_13(data_out3_13),
   .data_out3_14(data_out3_14),
   .data_out3_15(data_out3_15),
   .data_out3_16(data_out3_16),
   .data_out3_17(data_out3_17),
   .data_out3_18(data_out3_18),
   .data_out3_19(data_out3_19),
   .data_out3_20(data_out3_20),
   .data_out3_21(data_out3_21),
   .data_out3_22(data_out3_22),
   .data_out3_23(data_out3_23),
   .data_out3_24(data_out3_24),
   .conv_out_calc(conv_out_2),
   .valid_out_calc(valid_out_calc_2)
);

conv2_calc_3 conv2_calc_3(
   .clk(clk),
   .rst_n(rst_n),
   .valid_out_buf(valid_out_buf),
   .data_out1_0(data_out1_0),
   .data_out1_1(data_out1_1),
   .data_out1_2(data_out1_2),
   .data_out1_3(data_out1_3),
   .data_out1_4(data_out1_4),
   .data_out1_5(data_out1_5),
   .data_out1_6(data_out1_6),
   .data_out1_7(data_out1_7),
   .data_out1_8(data_out1_8),
   .data_out1_9(data_out1_9),
   .data_out1_10(data_out1_10),
   .data_out1_11(data_out1_11),
   .data_out1_12(data_out1_12),
   .data_out1_13(data_out1_13),
   .data_out1_14(data_out1_14),
   .data_out1_15(data_out1_15),
   .data_out1_16(data_out1_16),
   .data_out1_17(data_out1_17),
   .data_out1_18(data_out1_18),
   .data_out1_19(data_out1_19),
   .data_out1_20(data_out1_20),
   .data_out1_21(data_out1_21),
   .data_out1_22(data_out1_22),
   .data_out1_23(data_out1_23),
   .data_out1_24(data_out1_24),
   .data_out2_0(data_out2_0),
   .data_out2_1(data_out2_1),
   .data_out2_2(data_out2_2),
   .data_out2_3(data_out2_3),
   .data_out2_4(data_out2_4),
   .data_out2_5(data_out2_5),
   .data_out2_6(data_out2_6),
   .data_out2_7(data_out2_7),
   .data_out2_8(data_out2_8),
   .data_out2_9(data_out2_9),
   .data_out2_10(data_out2_10),
   .data_out2_11(data_out2_11),
   .data_out2_12(data_out2_12),
   .data_out2_13(data_out2_13),
   .data_out2_14(data_out2_14),
   .data_out2_15(data_out2_15),
   .data_out2_16(data_out2_16),
   .data_out2_17(data_out2_17),
   .data_out2_18(data_out2_18),
   .data_out2_19(data_out2_19),
   .data_out2_20(data_out2_20),
   .data_out2_21(data_out2_21),
   .data_out2_22(data_out2_22),
   .data_out2_23(data_out2_23),
   .data_out2_24(data_out2_24),
   .data_out3_0(data_out3_0),
   .data_out3_1(data_out3_1),
   .data_out3_2(data_out3_2),
   .data_out3_3(data_out3_3),
   .data_out3_4(data_out3_4),
   .data_out3_5(data_out3_5),
   .data_out3_6(data_out3_6),
   .data_out3_7(data_out3_7),
   .data_out3_8(data_out3_8),
   .data_out3_9(data_out3_9),
   .data_out3_10(data_out3_10),
   .data_out3_11(data_out3_11),
   .data_out3_12(data_out3_12),
   .data_out3_13(data_out3_13),
   .data_out3_14(data_out3_14),
   .data_out3_15(data_out3_15),
   .data_out3_16(data_out3_16),
   .data_out3_17(data_out3_17),
   .data_out3_18(data_out3_18),
   .data_out3_19(data_out3_19),
   .data_out3_20(data_out3_20),
   .data_out3_21(data_out3_21),
   .data_out3_22(data_out3_22),
   .data_out3_23(data_out3_23),
   .data_out3_24(data_out3_24),
   .conv_out_calc(conv_out_3),
   .valid_out_calc(valid_out_calc_3)
);

// Constants must be unpacked with continuous assignments, not always @(*):
// with only constants on the right-hand side the implicit sensitivity list is
// empty, so the block would never run and the bias would stay X in sim.
genvar gi;
generate
    for(gi=0;gi<=2;gi=gi+1) begin : unpack_bias
        assign bias[gi]=B_2[(8*gi)+:8];
    end
endgenerate
 assign exp_bias[0] = (bias[0][7] == 1) ? {4'b1111, bias[0]} : {4'b0000, bias[0]};
 assign exp_bias[1] = (bias[1][7] == 1) ? {4'b1111, bias[1]} : {4'b0000, bias[1]};
 assign exp_bias[2] = (bias[2][7] == 1) ? {4'b1111, bias[2]} : {4'b0000, bias[2]};

 assign conv2_out_1 = conv_out_1[12:1];
 assign conv2_out_2 = conv_out_2[12:1];
 assign conv2_out_3 = conv_out_3[12:1];
 
 endmodule

/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : conv2_buf.v
 *  Written by : Kang, Bo Young
 *  Written on : Oct 13, 2021
 *  Version    : 21.2
 *  Design     : 2nd Convolution Layer for CNN MNIST dataset
 *               Input Buffer
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: conv2_buf
 *------------------------------------------------------------------*/
 module conv2_buf #(parameter WIDTH = 12, HEIGHT = 12, DATA_BITS = 12) (
   input clk,
   input rst_n,
   input valid_in,
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
     buf_idx <= 0;
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
   if(valid_in) begin
     buf_idx <= buf_idx + 1'b1;
     if(buf_idx == WIDTH * FILTER_SIZE - 1) begin // buffer size = 140 = 28(w) * 5(h)
       buf_idx <= 0;
     end

     buffer[buf_idx] <= data_in;  // data input

     // Wait until first 140 input data filled in buffer
     if(!state) begin
       if(buf_idx == WIDTH * FILTER_SIZE - 1) begin
         state <= 1;
       end
     end else begin // valid state
       w_idx <= w_idx + 1'b1; // move right

      if(w_idx == WIDTH - FILTER_SIZE + 1) begin
        valid_out_buf <= 1'b0;  // unvalid area
      end else if(w_idx == WIDTH - 1) begin
        buf_flag <= buf_flag + 1;
        if(buf_flag == FILTER_SIZE - 1) begin
          buf_flag <= 0;
        end

        w_idx <= 0;

        if(h_idx == HEIGHT - FILTER_SIZE) begin // done 1 input read -> 28 * 28
          h_idx <= 0;
          state <= 0;
        end
          h_idx <= h_idx + 1;

      end else if(w_idx == 0) begin
        valid_out_buf <= 1'b1;  // start valid area
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
 end
endmodule

/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : conv2_calc_1.v
 *  Written by : Kang, Bo Young
 *  Written on : Oct 14, 2021
 *  Version    : 21.2
 *  Design     : 2nd Convolution Layer for CNN MNIST dataset
 *               Convolution Sum Calculation - 1st Channel
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: conv2_calc_1
 *------------------------------------------------------------------*/

 module conv2_calc_1 (
	input clk,
  input rst_n,
  input valid_out_buf,
	input signed [11:0] data_out1_0, data_out1_1, data_out1_2, data_out1_3, data_out1_4,
	  data_out1_5, data_out1_6, data_out1_7, data_out1_8, data_out1_9,
	  data_out1_10, data_out1_11, data_out1_12, data_out1_13, data_out1_14,
	  data_out1_15, data_out1_16, data_out1_17, data_out1_18, data_out1_19,
	  data_out1_20, data_out1_21, data_out1_22, data_out1_23, data_out1_24,
	
	  data_out2_0, data_out2_1, data_out2_2, data_out2_3, data_out2_4,
	  data_out2_5, data_out2_6, data_out2_7, data_out2_8, data_out2_9,
	  data_out2_10, data_out2_11, data_out2_12, data_out2_13, data_out2_14,
	  data_out2_15, data_out2_16, data_out2_17, data_out2_18, data_out2_19,
	  data_out2_20, data_out2_21, data_out2_22, data_out2_23, data_out2_24,
	
	  data_out3_0, data_out3_1, data_out3_2, data_out3_3, data_out3_4,
	  data_out3_5, data_out3_6, data_out3_7, data_out3_8, data_out3_9,
	  data_out3_10, data_out3_11, data_out3_12, data_out3_13, data_out3_14,
	  data_out3_15, data_out3_16, data_out3_17, data_out3_18, data_out3_19,
	  data_out3_20, data_out3_21, data_out3_22, data_out3_23, data_out3_24,

	  output reg [13:0] conv_out_calc,
  	output reg valid_out_calc
);

 localparam [0:23] B_2 = 24'h15FFF6;
 localparam signed [11:0] EXP_BIAS = {{4{B_2[0]}}, B_2[0+:8]};

// ---------------------------------------------------------------------
//  Fused constant-coefficient multiply-accumulate.
//
//  The 75 products used to be three 25-term sums merged afterwards.  Even
//  as a balanced tree that left 25 constant-multiplier CPAs plus a carry
//  propagation at every tree node, which is why re-balancing alone did not
//  move the slack.  Every multiplier is now expanded into shift/invert
//  rows and ALL of them are compressed by a 3:2 carry-save tree; the only
//  carry propagation left is one Kogge-Stone prefix adder in stage 2.
//
//  Negative weights use |w| rather than the raw two's-complement pattern:
//  0xF9 (-7) has six set bits but |w| = 7 has three, so half the rows
//  disappear.  -x == ~x + 1, and all those +1s plus the bias fold into one
//  constant row.  Rows are enumerated at generation time, so no parameter
//  modules and no constant functions are needed.
// ---------------------------------------------------------------------
localparam NR00 = 181;
wire [NR00*20-1:0] LV00;
assign LV00[0*20 +: 20] = ~{{8{data_out1_0[11]}}, data_out1_0};
assign LV00[1*20 +: 20] = ~({{8{data_out1_0[11]}}, data_out1_0} << 1);
assign LV00[2*20 +: 20] = ~({{8{data_out1_0[11]}}, data_out1_0} << 3);
assign LV00[3*20 +: 20] = ~({{8{data_out1_1[11]}}, data_out1_1} << 1);
assign LV00[4*20 +: 20] = ({{8{data_out1_2[11]}}, data_out1_2} << 1);
assign LV00[5*20 +: 20] = ({{8{data_out1_2[11]}}, data_out1_2} << 3);
assign LV00[6*20 +: 20] = {{8{data_out1_3[11]}}, data_out1_3};
assign LV00[7*20 +: 20] = ({{8{data_out1_4[11]}}, data_out1_4} << 1);
assign LV00[8*20 +: 20] = ({{8{data_out1_4[11]}}, data_out1_4} << 2);
assign LV00[9*20 +: 20] = ({{8{data_out1_4[11]}}, data_out1_4} << 3);
assign LV00[10*20 +: 20] = ~{{8{data_out1_5[11]}}, data_out1_5};
assign LV00[11*20 +: 20] = ~({{8{data_out1_5[11]}}, data_out1_5} << 1);
assign LV00[12*20 +: 20] = ~{{8{data_out1_6[11]}}, data_out1_6};
assign LV00[13*20 +: 20] = ~({{8{data_out1_6[11]}}, data_out1_6} << 2);
assign LV00[14*20 +: 20] = ~{{8{data_out1_7[11]}}, data_out1_7};
assign LV00[15*20 +: 20] = ~({{8{data_out1_7[11]}}, data_out1_7} << 3);
assign LV00[16*20 +: 20] = ~{{8{data_out1_8[11]}}, data_out1_8};
assign LV00[17*20 +: 20] = ~({{8{data_out1_9[11]}}, data_out1_9} << 1);
assign LV00[18*20 +: 20] = ~({{8{data_out1_9[11]}}, data_out1_9} << 2);
assign LV00[19*20 +: 20] = ~({{8{data_out1_9[11]}}, data_out1_9} << 3);
assign LV00[20*20 +: 20] = ~{{8{data_out1_10[11]}}, data_out1_10};
assign LV00[21*20 +: 20] = ~({{8{data_out1_10[11]}}, data_out1_10} << 1);
assign LV00[22*20 +: 20] = ~({{8{data_out1_10[11]}}, data_out1_10} << 2);
assign LV00[23*20 +: 20] = ~{{8{data_out1_11[11]}}, data_out1_11};
assign LV00[24*20 +: 20] = ~({{8{data_out1_12[11]}}, data_out1_12} << 1);
assign LV00[25*20 +: 20] = ~({{8{data_out1_12[11]}}, data_out1_12} << 3);
assign LV00[26*20 +: 20] = ~({{8{data_out1_13[11]}}, data_out1_13} << 2);
assign LV00[27*20 +: 20] = ~({{8{data_out1_14[11]}}, data_out1_14} << 2);
assign LV00[28*20 +: 20] = ({{8{data_out1_15[11]}}, data_out1_15} << 4);
assign LV00[29*20 +: 20] = ~{{8{data_out1_16[11]}}, data_out1_16};
assign LV00[30*20 +: 20] = ~({{8{data_out1_16[11]}}, data_out1_16} << 3);
assign LV00[31*20 +: 20] = ~{{8{data_out1_17[11]}}, data_out1_17};
assign LV00[32*20 +: 20] = ~({{8{data_out1_17[11]}}, data_out1_17} << 1);
assign LV00[33*20 +: 20] = ~({{8{data_out1_18[11]}}, data_out1_18} << 2);
assign LV00[34*20 +: 20] = ~({{8{data_out1_18[11]}}, data_out1_18} << 3);
assign LV00[35*20 +: 20] = ({{8{data_out1_19[11]}}, data_out1_19} << 2);
assign LV00[36*20 +: 20] = ~{{8{data_out1_20[11]}}, data_out1_20};
assign LV00[37*20 +: 20] = ~({{8{data_out1_20[11]}}, data_out1_20} << 3);
assign LV00[38*20 +: 20] = ~({{8{data_out1_21[11]}}, data_out1_21} << 2);
assign LV00[39*20 +: 20] = ~({{8{data_out1_21[11]}}, data_out1_21} << 3);
assign LV00[40*20 +: 20] = ({{8{data_out1_22[11]}}, data_out1_22} << 3);
assign LV00[41*20 +: 20] = ~({{8{data_out1_23[11]}}, data_out1_23} << 1);
assign LV00[42*20 +: 20] = ~({{8{data_out1_23[11]}}, data_out1_23} << 2);
assign LV00[43*20 +: 20] = ~({{8{data_out1_24[11]}}, data_out1_24} << 3);
assign LV00[44*20 +: 20] = ({{8{data_out2_0[11]}}, data_out2_0} << 2);
assign LV00[45*20 +: 20] = ({{8{data_out2_0[11]}}, data_out2_0} << 4);
assign LV00[46*20 +: 20] = ({{8{data_out2_0[11]}}, data_out2_0} << 5);
assign LV00[47*20 +: 20] = ({{8{data_out2_1[11]}}, data_out2_1} << 2);
assign LV00[48*20 +: 20] = ({{8{data_out2_1[11]}}, data_out2_1} << 3);
assign LV00[49*20 +: 20] = ~{{8{data_out2_2[11]}}, data_out2_2};
assign LV00[50*20 +: 20] = ~({{8{data_out2_2[11]}}, data_out2_2} << 1);
assign LV00[51*20 +: 20] = ~({{8{data_out2_2[11]}}, data_out2_2} << 2);
assign LV00[52*20 +: 20] = ~({{8{data_out2_3[11]}}, data_out2_3} << 1);
assign LV00[53*20 +: 20] = ~({{8{data_out2_3[11]}}, data_out2_3} << 3);
assign LV00[54*20 +: 20] = ~({{8{data_out2_3[11]}}, data_out2_3} << 6);
assign LV00[55*20 +: 20] = ~{{8{data_out2_4[11]}}, data_out2_4};
assign LV00[56*20 +: 20] = ~({{8{data_out2_4[11]}}, data_out2_4} << 2);
assign LV00[57*20 +: 20] = ~({{8{data_out2_4[11]}}, data_out2_4} << 4);
assign LV00[58*20 +: 20] = {{8{data_out2_5[11]}}, data_out2_5};
assign LV00[59*20 +: 20] = ({{8{data_out2_5[11]}}, data_out2_5} << 2);
assign LV00[60*20 +: 20] = ({{8{data_out2_5[11]}}, data_out2_5} << 3);
assign LV00[61*20 +: 20] = ({{8{data_out2_5[11]}}, data_out2_5} << 5);
assign LV00[62*20 +: 20] = ({{8{data_out2_6[11]}}, data_out2_6} << 1);
assign LV00[63*20 +: 20] = ({{8{data_out2_6[11]}}, data_out2_6} << 3);
assign LV00[64*20 +: 20] = ({{8{data_out2_6[11]}}, data_out2_6} << 4);
assign LV00[65*20 +: 20] = ({{8{data_out2_6[11]}}, data_out2_6} << 5);
assign LV00[66*20 +: 20] = ~{{8{data_out2_7[11]}}, data_out2_7};
assign LV00[67*20 +: 20] = ~({{8{data_out2_7[11]}}, data_out2_7} << 1);
assign LV00[68*20 +: 20] = ~({{8{data_out2_7[11]}}, data_out2_7} << 5);
assign LV00[69*20 +: 20] = ~({{8{data_out2_8[11]}}, data_out2_8} << 1);
assign LV00[70*20 +: 20] = ~({{8{data_out2_8[11]}}, data_out2_8} << 3);
assign LV00[71*20 +: 20] = ~({{8{data_out2_8[11]}}, data_out2_8} << 6);
assign LV00[72*20 +: 20] = {{8{data_out2_9[11]}}, data_out2_9};
assign LV00[73*20 +: 20] = ({{8{data_out2_9[11]}}, data_out2_9} << 1);
assign LV00[74*20 +: 20] = ({{8{data_out2_9[11]}}, data_out2_9} << 3);
assign LV00[75*20 +: 20] = ({{8{data_out2_9[11]}}, data_out2_9} << 4);
assign LV00[76*20 +: 20] = ({{8{data_out2_9[11]}}, data_out2_9} << 5);
assign LV00[77*20 +: 20] = ({{8{data_out2_10[11]}}, data_out2_10} << 1);
assign LV00[78*20 +: 20] = ({{8{data_out2_10[11]}}, data_out2_10} << 3);
assign LV00[79*20 +: 20] = ({{8{data_out2_10[11]}}, data_out2_10} << 6);
assign LV00[80*20 +: 20] = {{8{data_out2_11[11]}}, data_out2_11};
assign LV00[81*20 +: 20] = ({{8{data_out2_11[11]}}, data_out2_11} << 5);
assign LV00[82*20 +: 20] = ~({{8{data_out2_12[11]}}, data_out2_12} << 2);
assign LV00[83*20 +: 20] = ~({{8{data_out2_12[11]}}, data_out2_12} << 4);
assign LV00[84*20 +: 20] = ~({{8{data_out2_12[11]}}, data_out2_12} << 5);
assign LV00[85*20 +: 20] = ~({{8{data_out2_13[11]}}, data_out2_13} << 1);
assign LV00[86*20 +: 20] = ~({{8{data_out2_13[11]}}, data_out2_13} << 2);
assign LV00[87*20 +: 20] = ~({{8{data_out2_13[11]}}, data_out2_13} << 4);
assign LV00[88*20 +: 20] = ({{8{data_out2_14[11]}}, data_out2_14} << 2);
assign LV00[89*20 +: 20] = ({{8{data_out2_14[11]}}, data_out2_14} << 3);
assign LV00[90*20 +: 20] = ({{8{data_out2_14[11]}}, data_out2_14} << 4);
assign LV00[91*20 +: 20] = ({{8{data_out2_14[11]}}, data_out2_14} << 5);
assign LV00[92*20 +: 20] = ({{8{data_out2_15[11]}}, data_out2_15} << 1);
assign LV00[93*20 +: 20] = ({{8{data_out2_15[11]}}, data_out2_15} << 6);
assign LV00[94*20 +: 20] = ~{{8{data_out2_17[11]}}, data_out2_17};
assign LV00[95*20 +: 20] = ~({{8{data_out2_17[11]}}, data_out2_17} << 2);
assign LV00[96*20 +: 20] = ~({{8{data_out2_17[11]}}, data_out2_17} << 4);
assign LV00[97*20 +: 20] = ~({{8{data_out2_18[11]}}, data_out2_18} << 1);
assign LV00[98*20 +: 20] = ~({{8{data_out2_18[11]}}, data_out2_18} << 2);
assign LV00[99*20 +: 20] = ~({{8{data_out2_18[11]}}, data_out2_18} << 4);
assign LV00[100*20 +: 20] = ({{8{data_out2_19[11]}}, data_out2_19} << 3);
assign LV00[101*20 +: 20] = ({{8{data_out2_19[11]}}, data_out2_19} << 4);
assign LV00[102*20 +: 20] = ({{8{data_out2_20[11]}}, data_out2_20} << 3);
assign LV00[103*20 +: 20] = {{8{data_out2_21[11]}}, data_out2_21};
assign LV00[104*20 +: 20] = {{8{data_out2_22[11]}}, data_out2_22};
assign LV00[105*20 +: 20] = ({{8{data_out2_22[11]}}, data_out2_22} << 3);
assign LV00[106*20 +: 20] = ({{8{data_out2_22[11]}}, data_out2_22} << 4);
assign LV00[107*20 +: 20] = {{8{data_out2_23[11]}}, data_out2_23};
assign LV00[108*20 +: 20] = ({{8{data_out2_23[11]}}, data_out2_23} << 1);
assign LV00[109*20 +: 20] = ({{8{data_out2_24[11]}}, data_out2_24} << 3);
assign LV00[110*20 +: 20] = ({{8{data_out2_24[11]}}, data_out2_24} << 5);
assign LV00[111*20 +: 20] = ~{{8{data_out3_0[11]}}, data_out3_0};
assign LV00[112*20 +: 20] = ~({{8{data_out3_0[11]}}, data_out3_0} << 1);
assign LV00[113*20 +: 20] = ~{{8{data_out3_1[11]}}, data_out3_1};
assign LV00[114*20 +: 20] = ~{{8{data_out3_2[11]}}, data_out3_2};
assign LV00[115*20 +: 20] = ~({{8{data_out3_2[11]}}, data_out3_2} << 1);
assign LV00[116*20 +: 20] = ~({{8{data_out3_2[11]}}, data_out3_2} << 4);
assign LV00[117*20 +: 20] = ~({{8{data_out3_2[11]}}, data_out3_2} << 5);
assign LV00[118*20 +: 20] = ~({{8{data_out3_3[11]}}, data_out3_3} << 6);
assign LV00[119*20 +: 20] = ~{{8{data_out3_4[11]}}, data_out3_4};
assign LV00[120*20 +: 20] = ~({{8{data_out3_4[11]}}, data_out3_4} << 2);
assign LV00[121*20 +: 20] = ~({{8{data_out3_4[11]}}, data_out3_4} << 3);
assign LV00[122*20 +: 20] = ~{{8{data_out3_5[11]}}, data_out3_5};
assign LV00[123*20 +: 20] = ~({{8{data_out3_5[11]}}, data_out3_5} << 3);
assign LV00[124*20 +: 20] = ~({{8{data_out3_5[11]}}, data_out3_5} << 4);
assign LV00[125*20 +: 20] = ~({{8{data_out3_6[11]}}, data_out3_6} << 1);
assign LV00[126*20 +: 20] = ~({{8{data_out3_6[11]}}, data_out3_6} << 5);
assign LV00[127*20 +: 20] = ~{{8{data_out3_7[11]}}, data_out3_7};
assign LV00[128*20 +: 20] = ~({{8{data_out3_7[11]}}, data_out3_7} << 1);
assign LV00[129*20 +: 20] = ~({{8{data_out3_7[11]}}, data_out3_7} << 2);
assign LV00[130*20 +: 20] = ~({{8{data_out3_7[11]}}, data_out3_7} << 4);
assign LV00[131*20 +: 20] = ~({{8{data_out3_7[11]}}, data_out3_7} << 5);
assign LV00[132*20 +: 20] = ~({{8{data_out3_8[11]}}, data_out3_8} << 2);
assign LV00[133*20 +: 20] = ~({{8{data_out3_8[11]}}, data_out3_8} << 4);
assign LV00[134*20 +: 20] = ~({{8{data_out3_9[11]}}, data_out3_9} << 1);
assign LV00[135*20 +: 20] = ~{{8{data_out3_10[11]}}, data_out3_10};
assign LV00[136*20 +: 20] = ~({{8{data_out3_10[11]}}, data_out3_10} << 1);
assign LV00[137*20 +: 20] = ~({{8{data_out3_10[11]}}, data_out3_10} << 2);
assign LV00[138*20 +: 20] = ~({{8{data_out3_10[11]}}, data_out3_10} << 3);
assign LV00[139*20 +: 20] = ~({{8{data_out3_10[11]}}, data_out3_10} << 4);
assign LV00[140*20 +: 20] = ~({{8{data_out3_11[11]}}, data_out3_11} << 1);
assign LV00[141*20 +: 20] = ~({{8{data_out3_11[11]}}, data_out3_11} << 3);
assign LV00[142*20 +: 20] = ~({{8{data_out3_11[11]}}, data_out3_11} << 5);
assign LV00[143*20 +: 20] = ~({{8{data_out3_12[11]}}, data_out3_12} << 1);
assign LV00[144*20 +: 20] = ~{{8{data_out3_13[11]}}, data_out3_13};
assign LV00[145*20 +: 20] = ~({{8{data_out3_13[11]}}, data_out3_13} << 1);
assign LV00[146*20 +: 20] = ~({{8{data_out3_13[11]}}, data_out3_13} << 2);
assign LV00[147*20 +: 20] = ~({{8{data_out3_13[11]}}, data_out3_13} << 3);
assign LV00[148*20 +: 20] = ~({{8{data_out3_13[11]}}, data_out3_13} << 4);
assign LV00[149*20 +: 20] = ~({{8{data_out3_14[11]}}, data_out3_14} << 1);
assign LV00[150*20 +: 20] = ~({{8{data_out3_14[11]}}, data_out3_14} << 2);
assign LV00[151*20 +: 20] = ~({{8{data_out3_14[11]}}, data_out3_14} << 3);
assign LV00[152*20 +: 20] = ~({{8{data_out3_14[11]}}, data_out3_14} << 4);
assign LV00[153*20 +: 20] = ~({{8{data_out3_15[11]}}, data_out3_15} << 1);
assign LV00[154*20 +: 20] = ~({{8{data_out3_15[11]}}, data_out3_15} << 2);
assign LV00[155*20 +: 20] = ~({{8{data_out3_15[11]}}, data_out3_15} << 3);
assign LV00[156*20 +: 20] = ~({{8{data_out3_15[11]}}, data_out3_15} << 4);
assign LV00[157*20 +: 20] = ~{{8{data_out3_16[11]}}, data_out3_16};
assign LV00[158*20 +: 20] = ~({{8{data_out3_16[11]}}, data_out3_16} << 3);
assign LV00[159*20 +: 20] = ~{{8{data_out3_17[11]}}, data_out3_17};
assign LV00[160*20 +: 20] = ~({{8{data_out3_17[11]}}, data_out3_17} << 2);
assign LV00[161*20 +: 20] = ~({{8{data_out3_18[11]}}, data_out3_18} << 3);
assign LV00[162*20 +: 20] = ~({{8{data_out3_18[11]}}, data_out3_18} << 5);
assign LV00[163*20 +: 20] = ~{{8{data_out3_19[11]}}, data_out3_19};
assign LV00[164*20 +: 20] = ~({{8{data_out3_19[11]}}, data_out3_19} << 1);
assign LV00[165*20 +: 20] = ~({{8{data_out3_19[11]}}, data_out3_19} << 2);
assign LV00[166*20 +: 20] = ~{{8{data_out3_20[11]}}, data_out3_20};
assign LV00[167*20 +: 20] = ~({{8{data_out3_20[11]}}, data_out3_20} << 1);
assign LV00[168*20 +: 20] = ~({{8{data_out3_20[11]}}, data_out3_20} << 3);
assign LV00[169*20 +: 20] = ~({{8{data_out3_20[11]}}, data_out3_20} << 4);
assign LV00[170*20 +: 20] = {{8{data_out3_21[11]}}, data_out3_21};
assign LV00[171*20 +: 20] = ({{8{data_out3_21[11]}}, data_out3_21} << 2);
assign LV00[172*20 +: 20] = ({{8{data_out3_21[11]}}, data_out3_21} << 3);
assign LV00[173*20 +: 20] = ({{8{data_out3_21[11]}}, data_out3_21} << 4);
assign LV00[174*20 +: 20] = ({{8{data_out3_22[11]}}, data_out3_22} << 2);
assign LV00[175*20 +: 20] = ({{8{data_out3_22[11]}}, data_out3_22} << 3);
assign LV00[176*20 +: 20] = ~{{8{data_out3_23[11]}}, data_out3_23};
assign LV00[177*20 +: 20] = ~({{8{data_out3_23[11]}}, data_out3_23} << 2);
assign LV00[178*20 +: 20] = ({{8{data_out3_24[11]}}, data_out3_24} << 1);
assign LV00[179*20 +: 20] = ({{8{data_out3_24[11]}}, data_out3_24} << 2);
assign LV00[180*20 +: 20] = {{1{EXP_BIAS[11]}}, EXP_BIAS, 7'd0} + 20'd123;

// 3:2 compression : 181 -> 121 -> 81 -> 54 -> 36 -> 24 -> 16 -> 11 -> 8 -> 6 -> 4 -> 3 -> 2
// Pipeline cut after CSA07. CSA08+ and Kogge-Stone use registered MID07.
reg [11*20-1:0] MID07;
genvar cg;
generate
  wire [121*20-1:0] LV01;
  for (cg = 0; cg < 60; cg = cg + 1) begin : CSA01
    wire [19:0] a = LV00[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV00[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV00[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV01[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV01[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV01[120*20 +: 20] = LV00[180*20 +: 20];
  wire [81*20-1:0] LV02;
  for (cg = 0; cg < 40; cg = cg + 1) begin : CSA02
    wire [19:0] a = LV01[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV01[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV01[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV02[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV02[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV02[80*20 +: 20] = LV01[120*20 +: 20];
  wire [54*20-1:0] LV03;
  for (cg = 0; cg < 27; cg = cg + 1) begin : CSA03
    wire [19:0] a = LV02[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV02[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV02[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV03[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV03[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [36*20-1:0] LV04;
  for (cg = 0; cg < 18; cg = cg + 1) begin : CSA04
    wire [19:0] a = LV03[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV03[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV03[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV04[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV04[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [24*20-1:0] LV05;
  for (cg = 0; cg < 12; cg = cg + 1) begin : CSA05
    wire [19:0] a = LV04[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV04[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV04[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV05[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV05[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [16*20-1:0] LV06;
  for (cg = 0; cg < 8; cg = cg + 1) begin : CSA06
    wire [19:0] a = LV05[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV05[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV05[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV06[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV06[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [11*20-1:0] LV07;
  for (cg = 0; cg < 5; cg = cg + 1) begin : CSA07
    wire [19:0] a = LV06[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV06[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV06[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV07[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV07[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV07[10*20 +: 20] = LV06[15*20 +: 20];
  wire [8*20-1:0] LV08;
  for (cg = 0; cg < 3; cg = cg + 1) begin : CSA08
    wire [19:0] a = MID07[(3*cg+0)*20 +: 20];
    wire [19:0] b = MID07[(3*cg+1)*20 +: 20];
    wire [19:0] c = MID07[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV08[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV08[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV08[6*20 +: 20] = MID07[9*20 +: 20];
  assign LV08[7*20 +: 20] = MID07[10*20 +: 20];
  wire [6*20-1:0] LV09;
  for (cg = 0; cg < 2; cg = cg + 1) begin : CSA09
    wire [19:0] a = LV08[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV08[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV08[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV09[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV09[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV09[4*20 +: 20] = LV08[6*20 +: 20];
  assign LV09[5*20 +: 20] = LV08[7*20 +: 20];
  wire [4*20-1:0] LV10;
  for (cg = 0; cg < 2; cg = cg + 1) begin : CSA10
    wire [19:0] a = LV09[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV09[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV09[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV10[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV10[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [3*20-1:0] LV11;
  for (cg = 0; cg < 1; cg = cg + 1) begin : CSA11
    wire [19:0] a = LV10[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV10[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV10[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV11[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV11[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV11[2*20 +: 20] = LV10[3*20 +: 20];
  wire [2*20-1:0] LV12;
  for (cg = 0; cg < 1; cg = cg + 1) begin : CSA12
    wire [19:0] a = LV11[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV11[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV11[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV12[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV12[(2*cg+1)*20 +: 20] = cy << 1;
  end
endgenerate

reg vc_s1;

// Stage 2: remaining CSA (combo from MID07) plus the one Kogge-Stone add.
wire [19:0] ksg [0:5];
wire [19:0] ksp [0:5];
assign ksg[0] = LV12[0*20 +: 20] & LV12[1*20 +: 20];
assign ksp[0] = LV12[0*20 +: 20] ^ LV12[1*20 +: 20];
genvar kl, ki;
generate
  for (kl = 1; kl <= 5; kl = kl + 1) begin : KSL
    for (ki = 0; ki < 20; ki = ki + 1) begin : KSB
      if (ki >= (1 << (kl-1))) begin : UPD
        assign ksg[kl][ki] = ksg[kl-1][ki] | (ksp[kl-1][ki] & ksg[kl-1][ki-(1<<(kl-1))]);
        assign ksp[kl][ki] = ksp[kl-1][ki] & ksp[kl-1][ki-(1<<(kl-1))];
      end else begin : KEEP
        assign ksg[kl][ki] = ksg[kl-1][ki];
        assign ksp[kl][ki] = ksp[kl-1][ki];
      end
    end
  end
endgenerate
wire [19:0] ks_sum;
assign ks_sum[0] = ksp[0][0];
generate
  for (ki = 1; ki < 20; ki = ki + 1) begin : KSS
    assign ks_sum[ki] = ksp[0][ki] ^ ksg[5][ki-1];
  end
endgenerate

always @ (posedge clk) begin
	if(~rst_n) begin
		vc_s1 <= 0;
		valid_out_calc <= 0;
		conv_out_calc <= 0;
	end
	else begin
		MID07 <= LV07;
		// Toggling Valid Output Signal (unchanged sequence, on the stage-1 flag)
		if(valid_out_buf == 1) begin
			if(vc_s1 == 1)
				vc_s1 <= 0;
			else
				vc_s1 <= 1;
		end

		// stage 2 : resolve the carries and slice
		conv_out_calc  <= ks_sum[19:6];
		valid_out_calc <= vc_s1;
	end
end

endmodule


 module conv2_calc_2 (
	input clk,
  input rst_n,
  input valid_out_buf,
	input signed [11:0] data_out1_0, data_out1_1, data_out1_2, data_out1_3, data_out1_4,
	  data_out1_5, data_out1_6, data_out1_7, data_out1_8, data_out1_9,
	  data_out1_10, data_out1_11, data_out1_12, data_out1_13, data_out1_14,
	  data_out1_15, data_out1_16, data_out1_17, data_out1_18, data_out1_19,
	  data_out1_20, data_out1_21, data_out1_22, data_out1_23, data_out1_24,
	
	  data_out2_0, data_out2_1, data_out2_2, data_out2_3, data_out2_4,
	  data_out2_5, data_out2_6, data_out2_7, data_out2_8, data_out2_9,
	  data_out2_10, data_out2_11, data_out2_12, data_out2_13, data_out2_14,
	  data_out2_15, data_out2_16, data_out2_17, data_out2_18, data_out2_19,
	  data_out2_20, data_out2_21, data_out2_22, data_out2_23, data_out2_24,
	
	  data_out3_0, data_out3_1, data_out3_2, data_out3_3, data_out3_4,
	  data_out3_5, data_out3_6, data_out3_7, data_out3_8, data_out3_9,
	  data_out3_10, data_out3_11, data_out3_12, data_out3_13, data_out3_14,
	  data_out3_15, data_out3_16, data_out3_17, data_out3_18, data_out3_19,
	  data_out3_20, data_out3_21, data_out3_22, data_out3_23, data_out3_24,

	  output reg [13:0] conv_out_calc,
  	output reg valid_out_calc
);

 localparam [0:23] B_2 = 24'h15FFF6;
 localparam signed [11:0] EXP_BIAS = {{4{B_2[8]}}, B_2[8+:8]};

// ---------------------------------------------------------------------
//  Fused constant-coefficient multiply-accumulate.
//
//  The 75 products used to be three 25-term sums merged afterwards.  Even
//  as a balanced tree that left 25 constant-multiplier CPAs plus a carry
//  propagation at every tree node, which is why re-balancing alone did not
//  move the slack.  Every multiplier is now expanded into shift/invert
//  rows and ALL of them are compressed by a 3:2 carry-save tree; the only
//  carry propagation left is one Kogge-Stone prefix adder in stage 2.
//
//  Negative weights use |w| rather than the raw two's-complement pattern:
//  0xF9 (-7) has six set bits but |w| = 7 has three, so half the rows
//  disappear.  -x == ~x + 1, and all those +1s plus the bias fold into one
//  constant row.  Rows are enumerated at generation time, so no parameter
//  modules and no constant functions are needed.
// ---------------------------------------------------------------------
localparam NR00 = 183;
wire [NR00*20-1:0] LV00;
assign LV00[0*20 +: 20] = ~({{8{data_out1_0[11]}}, data_out1_0} << 1);
assign LV00[1*20 +: 20] = ~({{8{data_out1_0[11]}}, data_out1_0} << 3);
assign LV00[2*20 +: 20] = ~{{8{data_out1_1[11]}}, data_out1_1};
assign LV00[3*20 +: 20] = ~({{8{data_out1_1[11]}}, data_out1_1} << 3);
assign LV00[4*20 +: 20] = ({{8{data_out1_2[11]}}, data_out1_2} << 2);
assign LV00[5*20 +: 20] = {{8{data_out1_3[11]}}, data_out1_3};
assign LV00[6*20 +: 20] = ({{8{data_out1_3[11]}}, data_out1_3} << 2);
assign LV00[7*20 +: 20] = {{8{data_out1_4[11]}}, data_out1_4};
assign LV00[8*20 +: 20] = ({{8{data_out1_4[11]}}, data_out1_4} << 1);
assign LV00[9*20 +: 20] = ({{8{data_out1_4[11]}}, data_out1_4} << 2);
assign LV00[10*20 +: 20] = ({{8{data_out1_5[11]}}, data_out1_5} << 1);
assign LV00[11*20 +: 20] = ({{8{data_out1_5[11]}}, data_out1_5} << 2);
assign LV00[12*20 +: 20] = ({{8{data_out1_6[11]}}, data_out1_6} << 1);
assign LV00[13*20 +: 20] = ({{8{data_out1_6[11]}}, data_out1_6} << 2);
assign LV00[14*20 +: 20] = ({{8{data_out1_7[11]}}, data_out1_7} << 1);
assign LV00[15*20 +: 20] = ({{8{data_out1_7[11]}}, data_out1_7} << 2);
assign LV00[16*20 +: 20] = ~({{8{data_out1_8[11]}}, data_out1_8} << 2);
assign LV00[17*20 +: 20] = ~{{8{data_out1_9[11]}}, data_out1_9};
assign LV00[18*20 +: 20] = ~({{8{data_out1_9[11]}}, data_out1_9} << 1);
assign LV00[19*20 +: 20] = ~({{8{data_out1_9[11]}}, data_out1_9} << 2);
assign LV00[20*20 +: 20] = ~({{8{data_out1_10[11]}}, data_out1_10} << 2);
assign LV00[21*20 +: 20] = {{8{data_out1_11[11]}}, data_out1_11};
assign LV00[22*20 +: 20] = ({{8{data_out1_11[11]}}, data_out1_11} << 2);
assign LV00[23*20 +: 20] = ~{{8{data_out1_12[11]}}, data_out1_12};
assign LV00[24*20 +: 20] = ~({{8{data_out1_12[11]}}, data_out1_12} << 3);
assign LV00[25*20 +: 20] = {{8{data_out1_13[11]}}, data_out1_13};
assign LV00[26*20 +: 20] = ({{8{data_out1_13[11]}}, data_out1_13} << 3);
assign LV00[27*20 +: 20] = {{8{data_out1_14[11]}}, data_out1_14};
assign LV00[28*20 +: 20] = ({{8{data_out1_14[11]}}, data_out1_14} << 3);
assign LV00[29*20 +: 20] = ({{8{data_out1_15[11]}}, data_out1_15} << 1);
assign LV00[30*20 +: 20] = ({{8{data_out1_16[11]}}, data_out1_16} << 1);
assign LV00[31*20 +: 20] = ({{8{data_out1_16[11]}}, data_out1_16} << 2);
assign LV00[32*20 +: 20] = ({{8{data_out1_16[11]}}, data_out1_16} << 3);
assign LV00[33*20 +: 20] = ~{{8{data_out1_17[11]}}, data_out1_17};
assign LV00[34*20 +: 20] = ~({{8{data_out1_17[11]}}, data_out1_17} << 3);
assign LV00[35*20 +: 20] = ~{{8{data_out1_18[11]}}, data_out1_18};
assign LV00[36*20 +: 20] = ~({{8{data_out1_18[11]}}, data_out1_18} << 2);
assign LV00[37*20 +: 20] = ~({{8{data_out1_18[11]}}, data_out1_18} << 3);
assign LV00[38*20 +: 20] = {{8{data_out1_19[11]}}, data_out1_19};
assign LV00[39*20 +: 20] = {{8{data_out1_20[11]}}, data_out1_20};
assign LV00[40*20 +: 20] = ({{8{data_out1_20[11]}}, data_out1_20} << 1);
assign LV00[41*20 +: 20] = ({{8{data_out1_20[11]}}, data_out1_20} << 2);
assign LV00[42*20 +: 20] = ({{8{data_out1_21[11]}}, data_out1_21} << 1);
assign LV00[43*20 +: 20] = ~({{8{data_out1_22[11]}}, data_out1_22} << 3);
assign LV00[44*20 +: 20] = {{8{data_out1_24[11]}}, data_out1_24};
assign LV00[45*20 +: 20] = ({{8{data_out1_24[11]}}, data_out1_24} << 2);
assign LV00[46*20 +: 20] = ~({{8{data_out2_0[11]}}, data_out2_0} << 2);
assign LV00[47*20 +: 20] = ~({{8{data_out2_0[11]}}, data_out2_0} << 5);
assign LV00[48*20 +: 20] = ~({{8{data_out2_1[11]}}, data_out2_1} << 2);
assign LV00[49*20 +: 20] = ~({{8{data_out2_1[11]}}, data_out2_1} << 3);
assign LV00[50*20 +: 20] = ~({{8{data_out2_1[11]}}, data_out2_1} << 5);
assign LV00[51*20 +: 20] = ~{{8{data_out2_2[11]}}, data_out2_2};
assign LV00[52*20 +: 20] = ~({{8{data_out2_2[11]}}, data_out2_2} << 1);
assign LV00[53*20 +: 20] = ~({{8{data_out2_2[11]}}, data_out2_2} << 2);
assign LV00[54*20 +: 20] = ~({{8{data_out2_2[11]}}, data_out2_2} << 3);
assign LV00[55*20 +: 20] = ~{{8{data_out2_3[11]}}, data_out2_3};
assign LV00[56*20 +: 20] = ~({{8{data_out2_3[11]}}, data_out2_3} << 4);
assign LV00[57*20 +: 20] = {{8{data_out2_4[11]}}, data_out2_4};
assign LV00[58*20 +: 20] = ({{8{data_out2_4[11]}}, data_out2_4} << 3);
assign LV00[59*20 +: 20] = ({{8{data_out2_5[11]}}, data_out2_5} << 3);
assign LV00[60*20 +: 20] = ~{{8{data_out2_6[11]}}, data_out2_6};
assign LV00[61*20 +: 20] = ~({{8{data_out2_6[11]}}, data_out2_6} << 2);
assign LV00[62*20 +: 20] = ~({{8{data_out2_6[11]}}, data_out2_6} << 3);
assign LV00[63*20 +: 20] = ~({{8{data_out2_7[11]}}, data_out2_7} << 3);
assign LV00[64*20 +: 20] = {{8{data_out2_8[11]}}, data_out2_8};
assign LV00[65*20 +: 20] = ({{8{data_out2_8[11]}}, data_out2_8} << 2);
assign LV00[66*20 +: 20] = ({{8{data_out2_8[11]}}, data_out2_8} << 3);
assign LV00[67*20 +: 20] = ~{{8{data_out2_9[11]}}, data_out2_9};
assign LV00[68*20 +: 20] = ~({{8{data_out2_9[11]}}, data_out2_9} << 2);
assign LV00[69*20 +: 20] = ({{8{data_out2_10[11]}}, data_out2_10} << 2);
assign LV00[70*20 +: 20] = ({{8{data_out2_10[11]}}, data_out2_10} << 4);
assign LV00[71*20 +: 20] = ~{{8{data_out2_11[11]}}, data_out2_11};
assign LV00[72*20 +: 20] = ~({{8{data_out2_11[11]}}, data_out2_11} << 3);
assign LV00[73*20 +: 20] = ~{{8{data_out2_12[11]}}, data_out2_12};
assign LV00[74*20 +: 20] = ~({{8{data_out2_12[11]}}, data_out2_12} << 1);
assign LV00[75*20 +: 20] = ~({{8{data_out2_12[11]}}, data_out2_12} << 2);
assign LV00[76*20 +: 20] = ~({{8{data_out2_12[11]}}, data_out2_12} << 3);
assign LV00[77*20 +: 20] = ~({{8{data_out2_13[11]}}, data_out2_13} << 1);
assign LV00[78*20 +: 20] = ~({{8{data_out2_13[11]}}, data_out2_13} << 4);
assign LV00[79*20 +: 20] = ~({{8{data_out2_14[11]}}, data_out2_14} << 1);
assign LV00[80*20 +: 20] = ~({{8{data_out2_14[11]}}, data_out2_14} << 2);
assign LV00[81*20 +: 20] = ~({{8{data_out2_14[11]}}, data_out2_14} << 5);
assign LV00[82*20 +: 20] = ({{8{data_out2_15[11]}}, data_out2_15} << 3);
assign LV00[83*20 +: 20] = {{8{data_out2_16[11]}}, data_out2_16};
assign LV00[84*20 +: 20] = ({{8{data_out2_16[11]}}, data_out2_16} << 1);
assign LV00[85*20 +: 20] = ({{8{data_out2_16[11]}}, data_out2_16} << 3);
assign LV00[86*20 +: 20] = ({{8{data_out2_16[11]}}, data_out2_16} << 4);
assign LV00[87*20 +: 20] = ~{{8{data_out2_17[11]}}, data_out2_17};
assign LV00[88*20 +: 20] = ~({{8{data_out2_17[11]}}, data_out2_17} << 3);
assign LV00[89*20 +: 20] = ~({{8{data_out2_17[11]}}, data_out2_17} << 5);
assign LV00[90*20 +: 20] = ~{{8{data_out2_18[11]}}, data_out2_18};
assign LV00[91*20 +: 20] = ~({{8{data_out2_18[11]}}, data_out2_18} << 1);
assign LV00[92*20 +: 20] = ~({{8{data_out2_18[11]}}, data_out2_18} << 3);
assign LV00[93*20 +: 20] = ~({{8{data_out2_18[11]}}, data_out2_18} << 4);
assign LV00[94*20 +: 20] = ~{{8{data_out2_19[11]}}, data_out2_19};
assign LV00[95*20 +: 20] = ~({{8{data_out2_19[11]}}, data_out2_19} << 1);
assign LV00[96*20 +: 20] = ~({{8{data_out2_19[11]}}, data_out2_19} << 2);
assign LV00[97*20 +: 20] = ~{{8{data_out2_20[11]}}, data_out2_20};
assign LV00[98*20 +: 20] = ~({{8{data_out2_20[11]}}, data_out2_20} << 1);
assign LV00[99*20 +: 20] = ~({{8{data_out2_20[11]}}, data_out2_20} << 2);
assign LV00[100*20 +: 20] = ({{8{data_out2_21[11]}}, data_out2_21} << 1);
assign LV00[101*20 +: 20] = ({{8{data_out2_21[11]}}, data_out2_21} << 2);
assign LV00[102*20 +: 20] = ({{8{data_out2_21[11]}}, data_out2_21} << 3);
assign LV00[103*20 +: 20] = ({{8{data_out2_21[11]}}, data_out2_21} << 4);
assign LV00[104*20 +: 20] = {{8{data_out2_22[11]}}, data_out2_22};
assign LV00[105*20 +: 20] = ({{8{data_out2_22[11]}}, data_out2_22} << 3);
assign LV00[106*20 +: 20] = ({{8{data_out2_22[11]}}, data_out2_22} << 4);
assign LV00[107*20 +: 20] = ({{8{data_out2_23[11]}}, data_out2_23} << 4);
assign LV00[108*20 +: 20] = ({{8{data_out2_23[11]}}, data_out2_23} << 5);
assign LV00[109*20 +: 20] = ({{8{data_out2_24[11]}}, data_out2_24} << 1);
assign LV00[110*20 +: 20] = ({{8{data_out2_24[11]}}, data_out2_24} << 2);
assign LV00[111*20 +: 20] = ({{8{data_out2_24[11]}}, data_out2_24} << 4);
assign LV00[112*20 +: 20] = ~{{8{data_out3_0[11]}}, data_out3_0};
assign LV00[113*20 +: 20] = ~({{8{data_out3_0[11]}}, data_out3_0} << 2);
assign LV00[114*20 +: 20] = ~({{8{data_out3_0[11]}}, data_out3_0} << 4);
assign LV00[115*20 +: 20] = ~({{8{data_out3_1[11]}}, data_out3_1} << 2);
assign LV00[116*20 +: 20] = ~({{8{data_out3_1[11]}}, data_out3_1} << 3);
assign LV00[117*20 +: 20] = ~({{8{data_out3_2[11]}}, data_out3_2} << 2);
assign LV00[118*20 +: 20] = ~({{8{data_out3_2[11]}}, data_out3_2} << 4);
assign LV00[119*20 +: 20] = {{8{data_out3_4[11]}}, data_out3_4};
assign LV00[120*20 +: 20] = ({{8{data_out3_4[11]}}, data_out3_4} << 1);
assign LV00[121*20 +: 20] = ({{8{data_out3_4[11]}}, data_out3_4} << 3);
assign LV00[122*20 +: 20] = ({{8{data_out3_4[11]}}, data_out3_4} << 4);
assign LV00[123*20 +: 20] = {{8{data_out3_5[11]}}, data_out3_5};
assign LV00[124*20 +: 20] = ({{8{data_out3_5[11]}}, data_out3_5} << 3);
assign LV00[125*20 +: 20] = ({{8{data_out3_5[11]}}, data_out3_5} << 4);
assign LV00[126*20 +: 20] = {{8{data_out3_6[11]}}, data_out3_6};
assign LV00[127*20 +: 20] = ({{8{data_out3_6[11]}}, data_out3_6} << 1);
assign LV00[128*20 +: 20] = ({{8{data_out3_6[11]}}, data_out3_6} << 2);
assign LV00[129*20 +: 20] = ({{8{data_out3_6[11]}}, data_out3_6} << 3);
assign LV00[130*20 +: 20] = ~({{8{data_out3_7[11]}}, data_out3_7} << 3);
assign LV00[131*20 +: 20] = ({{8{data_out3_8[11]}}, data_out3_8} << 1);
assign LV00[132*20 +: 20] = ({{8{data_out3_8[11]}}, data_out3_8} << 2);
assign LV00[133*20 +: 20] = ({{8{data_out3_8[11]}}, data_out3_8} << 3);
assign LV00[134*20 +: 20] = ({{8{data_out3_8[11]}}, data_out3_8} << 4);
assign LV00[135*20 +: 20] = ({{8{data_out3_9[11]}}, data_out3_9} << 1);
assign LV00[136*20 +: 20] = ({{8{data_out3_9[11]}}, data_out3_9} << 2);
assign LV00[137*20 +: 20] = ({{8{data_out3_9[11]}}, data_out3_9} << 5);
assign LV00[138*20 +: 20] = {{8{data_out3_10[11]}}, data_out3_10};
assign LV00[139*20 +: 20] = ({{8{data_out3_10[11]}}, data_out3_10} << 1);
assign LV00[140*20 +: 20] = ({{8{data_out3_11[11]}}, data_out3_11} << 3);
assign LV00[141*20 +: 20] = ({{8{data_out3_11[11]}}, data_out3_11} << 4);
assign LV00[142*20 +: 20] = {{8{data_out3_12[11]}}, data_out3_12};
assign LV00[143*20 +: 20] = ({{8{data_out3_12[11]}}, data_out3_12} << 3);
assign LV00[144*20 +: 20] = ({{8{data_out3_12[11]}}, data_out3_12} << 4);
assign LV00[145*20 +: 20] = {{8{data_out3_13[11]}}, data_out3_13};
assign LV00[146*20 +: 20] = ({{8{data_out3_13[11]}}, data_out3_13} << 1);
assign LV00[147*20 +: 20] = ({{8{data_out3_13[11]}}, data_out3_13} << 2);
assign LV00[148*20 +: 20] = ({{8{data_out3_13[11]}}, data_out3_13} << 3);
assign LV00[149*20 +: 20] = {{8{data_out3_14[11]}}, data_out3_14};
assign LV00[150*20 +: 20] = ({{8{data_out3_14[11]}}, data_out3_14} << 3);
assign LV00[151*20 +: 20] = ~{{8{data_out3_15[11]}}, data_out3_15};
assign LV00[152*20 +: 20] = ~({{8{data_out3_15[11]}}, data_out3_15} << 1);
assign LV00[153*20 +: 20] = ~({{8{data_out3_15[11]}}, data_out3_15} << 2);
assign LV00[154*20 +: 20] = {{8{data_out3_16[11]}}, data_out3_16};
assign LV00[155*20 +: 20] = ({{8{data_out3_16[11]}}, data_out3_16} << 2);
assign LV00[156*20 +: 20] = ({{8{data_out3_16[11]}}, data_out3_16} << 3);
assign LV00[157*20 +: 20] = ({{8{data_out3_16[11]}}, data_out3_16} << 4);
assign LV00[158*20 +: 20] = ({{8{data_out3_17[11]}}, data_out3_17} << 1);
assign LV00[159*20 +: 20] = ({{8{data_out3_17[11]}}, data_out3_17} << 3);
assign LV00[160*20 +: 20] = ({{8{data_out3_17[11]}}, data_out3_17} << 5);
assign LV00[161*20 +: 20] = ({{8{data_out3_18[11]}}, data_out3_18} << 1);
assign LV00[162*20 +: 20] = ({{8{data_out3_18[11]}}, data_out3_18} << 3);
assign LV00[163*20 +: 20] = ({{8{data_out3_18[11]}}, data_out3_18} << 4);
assign LV00[164*20 +: 20] = ({{8{data_out3_19[11]}}, data_out3_19} << 3);
assign LV00[165*20 +: 20] = ({{8{data_out3_19[11]}}, data_out3_19} << 4);
assign LV00[166*20 +: 20] = ~({{8{data_out3_20[11]}}, data_out3_20} << 1);
assign LV00[167*20 +: 20] = ~({{8{data_out3_20[11]}}, data_out3_20} << 2);
assign LV00[168*20 +: 20] = ~({{8{data_out3_20[11]}}, data_out3_20} << 4);
assign LV00[169*20 +: 20] = ({{8{data_out3_21[11]}}, data_out3_21} << 3);
assign LV00[170*20 +: 20] = ({{8{data_out3_21[11]}}, data_out3_21} << 4);
assign LV00[171*20 +: 20] = ({{8{data_out3_21[11]}}, data_out3_21} << 5);
assign LV00[172*20 +: 20] = ({{8{data_out3_22[11]}}, data_out3_22} << 1);
assign LV00[173*20 +: 20] = ({{8{data_out3_22[11]}}, data_out3_22} << 4);
assign LV00[174*20 +: 20] = ({{8{data_out3_22[11]}}, data_out3_22} << 6);
assign LV00[175*20 +: 20] = {{8{data_out3_23[11]}}, data_out3_23};
assign LV00[176*20 +: 20] = ({{8{data_out3_23[11]}}, data_out3_23} << 1);
assign LV00[177*20 +: 20] = ({{8{data_out3_23[11]}}, data_out3_23} << 2);
assign LV00[178*20 +: 20] = ({{8{data_out3_23[11]}}, data_out3_23} << 6);
assign LV00[179*20 +: 20] = {{8{data_out3_24[11]}}, data_out3_24};
assign LV00[180*20 +: 20] = ({{8{data_out3_24[11]}}, data_out3_24} << 2);
assign LV00[181*20 +: 20] = ({{8{data_out3_24[11]}}, data_out3_24} << 3);
assign LV00[182*20 +: 20] = {{1{EXP_BIAS[11]}}, EXP_BIAS, 7'd0} + 20'd72;

// 3:2 compression : 183 -> 122 -> 82 -> 55 -> 37 -> 25 -> 17 -> 12 -> 8 -> 6 -> 4 -> 3 -> 2
// Pipeline cut after CSA07. CSA08+ and Kogge-Stone use registered MID07.
reg [12*20-1:0] MID07;
genvar cg;
generate
  wire [122*20-1:0] LV01;
  for (cg = 0; cg < 61; cg = cg + 1) begin : CSA01
    wire [19:0] a = LV00[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV00[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV00[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV01[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV01[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [82*20-1:0] LV02;
  for (cg = 0; cg < 40; cg = cg + 1) begin : CSA02
    wire [19:0] a = LV01[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV01[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV01[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV02[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV02[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV02[80*20 +: 20] = LV01[120*20 +: 20];
  assign LV02[81*20 +: 20] = LV01[121*20 +: 20];
  wire [55*20-1:0] LV03;
  for (cg = 0; cg < 27; cg = cg + 1) begin : CSA03
    wire [19:0] a = LV02[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV02[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV02[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV03[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV03[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV03[54*20 +: 20] = LV02[81*20 +: 20];
  wire [37*20-1:0] LV04;
  for (cg = 0; cg < 18; cg = cg + 1) begin : CSA04
    wire [19:0] a = LV03[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV03[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV03[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV04[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV04[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV04[36*20 +: 20] = LV03[54*20 +: 20];
  wire [25*20-1:0] LV05;
  for (cg = 0; cg < 12; cg = cg + 1) begin : CSA05
    wire [19:0] a = LV04[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV04[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV04[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV05[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV05[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV05[24*20 +: 20] = LV04[36*20 +: 20];
  wire [17*20-1:0] LV06;
  for (cg = 0; cg < 8; cg = cg + 1) begin : CSA06
    wire [19:0] a = LV05[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV05[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV05[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV06[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV06[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV06[16*20 +: 20] = LV05[24*20 +: 20];
  wire [12*20-1:0] LV07;
  for (cg = 0; cg < 5; cg = cg + 1) begin : CSA07
    wire [19:0] a = LV06[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV06[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV06[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV07[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV07[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV07[10*20 +: 20] = LV06[15*20 +: 20];
  assign LV07[11*20 +: 20] = LV06[16*20 +: 20];
  wire [8*20-1:0] LV08;
  for (cg = 0; cg < 4; cg = cg + 1) begin : CSA08
    wire [19:0] a = MID07[(3*cg+0)*20 +: 20];
    wire [19:0] b = MID07[(3*cg+1)*20 +: 20];
    wire [19:0] c = MID07[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV08[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV08[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [6*20-1:0] LV09;
  for (cg = 0; cg < 2; cg = cg + 1) begin : CSA09
    wire [19:0] a = LV08[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV08[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV08[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV09[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV09[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV09[4*20 +: 20] = LV08[6*20 +: 20];
  assign LV09[5*20 +: 20] = LV08[7*20 +: 20];
  wire [4*20-1:0] LV10;
  for (cg = 0; cg < 2; cg = cg + 1) begin : CSA10
    wire [19:0] a = LV09[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV09[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV09[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV10[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV10[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [3*20-1:0] LV11;
  for (cg = 0; cg < 1; cg = cg + 1) begin : CSA11
    wire [19:0] a = LV10[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV10[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV10[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV11[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV11[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV11[2*20 +: 20] = LV10[3*20 +: 20];
  wire [2*20-1:0] LV12;
  for (cg = 0; cg < 1; cg = cg + 1) begin : CSA12
    wire [19:0] a = LV11[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV11[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV11[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV12[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV12[(2*cg+1)*20 +: 20] = cy << 1;
  end
endgenerate

reg vc_s1;

// Stage 2: remaining CSA (combo from MID07) plus the one Kogge-Stone add.
wire [19:0] ksg [0:5];
wire [19:0] ksp [0:5];
assign ksg[0] = LV12[0*20 +: 20] & LV12[1*20 +: 20];
assign ksp[0] = LV12[0*20 +: 20] ^ LV12[1*20 +: 20];
genvar kl, ki;
generate
  for (kl = 1; kl <= 5; kl = kl + 1) begin : KSL
    for (ki = 0; ki < 20; ki = ki + 1) begin : KSB
      if (ki >= (1 << (kl-1))) begin : UPD
        assign ksg[kl][ki] = ksg[kl-1][ki] | (ksp[kl-1][ki] & ksg[kl-1][ki-(1<<(kl-1))]);
        assign ksp[kl][ki] = ksp[kl-1][ki] & ksp[kl-1][ki-(1<<(kl-1))];
      end else begin : KEEP
        assign ksg[kl][ki] = ksg[kl-1][ki];
        assign ksp[kl][ki] = ksp[kl-1][ki];
      end
    end
  end
endgenerate
wire [19:0] ks_sum;
assign ks_sum[0] = ksp[0][0];
generate
  for (ki = 1; ki < 20; ki = ki + 1) begin : KSS
    assign ks_sum[ki] = ksp[0][ki] ^ ksg[5][ki-1];
  end
endgenerate

always @ (posedge clk) begin
	if(~rst_n) begin
		vc_s1 <= 0;
		valid_out_calc <= 0;
		conv_out_calc <= 0;
	end
	else begin
		MID07 <= LV07;
		// Toggling Valid Output Signal (unchanged sequence, on the stage-1 flag)
		if(valid_out_buf == 1) begin
			if(vc_s1 == 1)
				vc_s1 <= 0;
			else
				vc_s1 <= 1;
		end

		// stage 2 : resolve the carries and slice
		conv_out_calc  <= ks_sum[19:6];
		valid_out_calc <= vc_s1;
	end
end

endmodule


 module conv2_calc_3 (
	input clk,
  input rst_n,
  input valid_out_buf,
	input signed [11:0] data_out1_0, data_out1_1, data_out1_2, data_out1_3, data_out1_4,
	  data_out1_5, data_out1_6, data_out1_7, data_out1_8, data_out1_9,
	  data_out1_10, data_out1_11, data_out1_12, data_out1_13, data_out1_14,
	  data_out1_15, data_out1_16, data_out1_17, data_out1_18, data_out1_19,
	  data_out1_20, data_out1_21, data_out1_22, data_out1_23, data_out1_24,
	
	  data_out2_0, data_out2_1, data_out2_2, data_out2_3, data_out2_4,
	  data_out2_5, data_out2_6, data_out2_7, data_out2_8, data_out2_9,
	  data_out2_10, data_out2_11, data_out2_12, data_out2_13, data_out2_14,
	  data_out2_15, data_out2_16, data_out2_17, data_out2_18, data_out2_19,
	  data_out2_20, data_out2_21, data_out2_22, data_out2_23, data_out2_24,
	
	  data_out3_0, data_out3_1, data_out3_2, data_out3_3, data_out3_4,
	  data_out3_5, data_out3_6, data_out3_7, data_out3_8, data_out3_9,
	  data_out3_10, data_out3_11, data_out3_12, data_out3_13, data_out3_14,
	  data_out3_15, data_out3_16, data_out3_17, data_out3_18, data_out3_19,
	  data_out3_20, data_out3_21, data_out3_22, data_out3_23, data_out3_24,

	  output reg [13:0] conv_out_calc,
  	output reg valid_out_calc
);

 localparam [0:23] B_2 = 24'h15FFF6;
 localparam signed [11:0] EXP_BIAS = {{4{B_2[16]}}, B_2[16+:8]};

// ---------------------------------------------------------------------
//  Fused constant-coefficient multiply-accumulate.
//
//  The 75 products used to be three 25-term sums merged afterwards.  Even
//  as a balanced tree that left 25 constant-multiplier CPAs plus a carry
//  propagation at every tree node, which is why re-balancing alone did not
//  move the slack.  Every multiplier is now expanded into shift/invert
//  rows and ALL of them are compressed by a 3:2 carry-save tree; the only
//  carry propagation left is one Kogge-Stone prefix adder in stage 2.
//
//  Negative weights use |w| rather than the raw two's-complement pattern:
//  0xF9 (-7) has six set bits but |w| = 7 has three, so half the rows
//  disappear.  -x == ~x + 1, and all those +1s plus the bias fold into one
//  constant row.  Rows are enumerated at generation time, so no parameter
//  modules and no constant functions are needed.
// ---------------------------------------------------------------------
localparam NR00 = 170;
wire [NR00*20-1:0] LV00;
assign LV00[0*20 +: 20] = ~{{8{data_out1_0[11]}}, data_out1_0};
assign LV00[1*20 +: 20] = ~({{8{data_out1_0[11]}}, data_out1_0} << 1);
assign LV00[2*20 +: 20] = {{8{data_out1_1[11]}}, data_out1_1};
assign LV00[3*20 +: 20] = ({{8{data_out1_1[11]}}, data_out1_1} << 1);
assign LV00[4*20 +: 20] = ({{8{data_out1_1[11]}}, data_out1_1} << 3);
assign LV00[5*20 +: 20] = ~({{8{data_out1_2[11]}}, data_out1_2} << 1);
assign LV00[6*20 +: 20] = ~({{8{data_out1_2[11]}}, data_out1_2} << 2);
assign LV00[7*20 +: 20] = ~{{8{data_out1_3[11]}}, data_out1_3};
assign LV00[8*20 +: 20] = ~({{8{data_out1_3[11]}}, data_out1_3} << 1);
assign LV00[9*20 +: 20] = ~({{8{data_out1_3[11]}}, data_out1_3} << 2);
assign LV00[10*20 +: 20] = ~({{8{data_out1_4[11]}}, data_out1_4} << 1);
assign LV00[11*20 +: 20] = {{8{data_out1_6[11]}}, data_out1_6};
assign LV00[12*20 +: 20] = ~({{8{data_out1_7[11]}}, data_out1_7} << 2);
assign LV00[13*20 +: 20] = ~({{8{data_out1_7[11]}}, data_out1_7} << 3);
assign LV00[14*20 +: 20] = ~{{8{data_out1_8[11]}}, data_out1_8};
assign LV00[15*20 +: 20] = ~({{8{data_out1_8[11]}}, data_out1_8} << 2);
assign LV00[16*20 +: 20] = ~({{8{data_out1_8[11]}}, data_out1_8} << 3);
assign LV00[17*20 +: 20] = {{8{data_out1_9[11]}}, data_out1_9};
assign LV00[18*20 +: 20] = ({{8{data_out1_9[11]}}, data_out1_9} << 1);
assign LV00[19*20 +: 20] = ~({{8{data_out1_10[11]}}, data_out1_10} << 3);
assign LV00[20*20 +: 20] = ~({{8{data_out1_11[11]}}, data_out1_11} << 2);
assign LV00[21*20 +: 20] = ~({{8{data_out1_11[11]}}, data_out1_11} << 3);
assign LV00[22*20 +: 20] = ({{8{data_out1_12[11]}}, data_out1_12} << 2);
assign LV00[23*20 +: 20] = ({{8{data_out1_12[11]}}, data_out1_12} << 3);
assign LV00[24*20 +: 20] = ~({{8{data_out1_13[11]}}, data_out1_13} << 1);
assign LV00[25*20 +: 20] = {{8{data_out1_14[11]}}, data_out1_14};
assign LV00[26*20 +: 20] = ({{8{data_out1_14[11]}}, data_out1_14} << 1);
assign LV00[27*20 +: 20] = ({{8{data_out1_14[11]}}, data_out1_14} << 2);
assign LV00[28*20 +: 20] = ~({{8{data_out1_15[11]}}, data_out1_15} << 2);
assign LV00[29*20 +: 20] = ~({{8{data_out1_15[11]}}, data_out1_15} << 3);
assign LV00[30*20 +: 20] = ~({{8{data_out1_16[11]}}, data_out1_16} << 2);
assign LV00[31*20 +: 20] = ({{8{data_out1_17[11]}}, data_out1_17} << 1);
assign LV00[32*20 +: 20] = ~{{8{data_out1_18[11]}}, data_out1_18};
assign LV00[33*20 +: 20] = ~({{8{data_out1_18[11]}}, data_out1_18} << 3);
assign LV00[34*20 +: 20] = ~({{8{data_out1_19[11]}}, data_out1_19} << 1);
assign LV00[35*20 +: 20] = ~({{8{data_out1_19[11]}}, data_out1_19} << 2);
assign LV00[36*20 +: 20] = ({{8{data_out1_20[11]}}, data_out1_20} << 2);
assign LV00[37*20 +: 20] = ({{8{data_out1_20[11]}}, data_out1_20} << 3);
assign LV00[38*20 +: 20] = ({{8{data_out1_21[11]}}, data_out1_21} << 3);
assign LV00[39*20 +: 20] = {{8{data_out1_22[11]}}, data_out1_22};
assign LV00[40*20 +: 20] = ({{8{data_out1_22[11]}}, data_out1_22} << 1);
assign LV00[41*20 +: 20] = ({{8{data_out1_22[11]}}, data_out1_22} << 2);
assign LV00[42*20 +: 20] = ({{8{data_out1_22[11]}}, data_out1_22} << 3);
assign LV00[43*20 +: 20] = ({{8{data_out1_23[11]}}, data_out1_23} << 1);
assign LV00[44*20 +: 20] = {{8{data_out1_24[11]}}, data_out1_24};
assign LV00[45*20 +: 20] = ({{8{data_out1_24[11]}}, data_out1_24} << 1);
assign LV00[46*20 +: 20] = ~{{8{data_out2_0[11]}}, data_out2_0};
assign LV00[47*20 +: 20] = ~({{8{data_out2_0[11]}}, data_out2_0} << 4);
assign LV00[48*20 +: 20] = {{8{data_out2_1[11]}}, data_out2_1};
assign LV00[49*20 +: 20] = ({{8{data_out2_1[11]}}, data_out2_1} << 1);
assign LV00[50*20 +: 20] = ({{8{data_out2_1[11]}}, data_out2_1} << 2);
assign LV00[51*20 +: 20] = ({{8{data_out2_2[11]}}, data_out2_2} << 1);
assign LV00[52*20 +: 20] = ({{8{data_out2_2[11]}}, data_out2_2} << 2);
assign LV00[53*20 +: 20] = ~{{8{data_out2_3[11]}}, data_out2_3};
assign LV00[54*20 +: 20] = ~({{8{data_out2_3[11]}}, data_out2_3} << 2);
assign LV00[55*20 +: 20] = ~({{8{data_out2_3[11]}}, data_out2_3} << 3);
assign LV00[56*20 +: 20] = {{8{data_out2_4[11]}}, data_out2_4};
assign LV00[57*20 +: 20] = ({{8{data_out2_4[11]}}, data_out2_4} << 2);
assign LV00[58*20 +: 20] = ({{8{data_out2_4[11]}}, data_out2_4} << 3);
assign LV00[59*20 +: 20] = ~({{8{data_out2_5[11]}}, data_out2_5} << 1);
assign LV00[60*20 +: 20] = ~({{8{data_out2_5[11]}}, data_out2_5} << 4);
assign LV00[61*20 +: 20] = ~({{8{data_out2_6[11]}}, data_out2_6} << 2);
assign LV00[62*20 +: 20] = ~({{8{data_out2_6[11]}}, data_out2_6} << 3);
assign LV00[63*20 +: 20] = ~({{8{data_out2_6[11]}}, data_out2_6} << 5);
assign LV00[64*20 +: 20] = ~{{8{data_out2_7[11]}}, data_out2_7};
assign LV00[65*20 +: 20] = ~({{8{data_out2_7[11]}}, data_out2_7} << 1);
assign LV00[66*20 +: 20] = ~({{8{data_out2_7[11]}}, data_out2_7} << 3);
assign LV00[67*20 +: 20] = ~({{8{data_out2_7[11]}}, data_out2_7} << 4);
assign LV00[68*20 +: 20] = ~({{8{data_out2_7[11]}}, data_out2_7} << 5);
assign LV00[69*20 +: 20] = ({{8{data_out2_8[11]}}, data_out2_8} << 2);
assign LV00[70*20 +: 20] = {{8{data_out2_9[11]}}, data_out2_9};
assign LV00[71*20 +: 20] = ({{8{data_out2_9[11]}}, data_out2_9} << 2);
assign LV00[72*20 +: 20] = ({{8{data_out2_9[11]}}, data_out2_9} << 4);
assign LV00[73*20 +: 20] = ({{8{data_out2_9[11]}}, data_out2_9} << 5);
assign LV00[74*20 +: 20] = ~{{8{data_out2_10[11]}}, data_out2_10};
assign LV00[75*20 +: 20] = ~({{8{data_out2_10[11]}}, data_out2_10} << 2);
assign LV00[76*20 +: 20] = ~({{8{data_out2_10[11]}}, data_out2_10} << 3);
assign LV00[77*20 +: 20] = ~{{8{data_out2_11[11]}}, data_out2_11};
assign LV00[78*20 +: 20] = ~({{8{data_out2_11[11]}}, data_out2_11} << 3);
assign LV00[79*20 +: 20] = ~({{8{data_out2_11[11]}}, data_out2_11} << 4);
assign LV00[80*20 +: 20] = {{8{data_out2_12[11]}}, data_out2_12};
assign LV00[81*20 +: 20] = ({{8{data_out2_12[11]}}, data_out2_12} << 3);
assign LV00[82*20 +: 20] = ({{8{data_out2_12[11]}}, data_out2_12} << 4);
assign LV00[83*20 +: 20] = {{8{data_out2_13[11]}}, data_out2_13};
assign LV00[84*20 +: 20] = ({{8{data_out2_13[11]}}, data_out2_13} << 3);
assign LV00[85*20 +: 20] = ({{8{data_out2_13[11]}}, data_out2_13} << 4);
assign LV00[86*20 +: 20] = ({{8{data_out2_13[11]}}, data_out2_13} << 5);
assign LV00[87*20 +: 20] = ({{8{data_out2_14[11]}}, data_out2_14} << 1);
assign LV00[88*20 +: 20] = ({{8{data_out2_14[11]}}, data_out2_14} << 2);
assign LV00[89*20 +: 20] = ({{8{data_out2_14[11]}}, data_out2_14} << 3);
assign LV00[90*20 +: 20] = ({{8{data_out2_14[11]}}, data_out2_14} << 5);
assign LV00[91*20 +: 20] = {{8{data_out2_15[11]}}, data_out2_15};
assign LV00[92*20 +: 20] = ({{8{data_out2_15[11]}}, data_out2_15} << 1);
assign LV00[93*20 +: 20] = ({{8{data_out2_15[11]}}, data_out2_15} << 4);
assign LV00[94*20 +: 20] = ({{8{data_out2_16[11]}}, data_out2_16} << 4);
assign LV00[95*20 +: 20] = {{8{data_out2_17[11]}}, data_out2_17};
assign LV00[96*20 +: 20] = ({{8{data_out2_17[11]}}, data_out2_17} << 1);
assign LV00[97*20 +: 20] = ({{8{data_out2_17[11]}}, data_out2_17} << 2);
assign LV00[98*20 +: 20] = ({{8{data_out2_17[11]}}, data_out2_17} << 4);
assign LV00[99*20 +: 20] = ({{8{data_out2_17[11]}}, data_out2_17} << 5);
assign LV00[100*20 +: 20] = {{8{data_out2_18[11]}}, data_out2_18};
assign LV00[101*20 +: 20] = ({{8{data_out2_18[11]}}, data_out2_18} << 1);
assign LV00[102*20 +: 20] = ({{8{data_out2_18[11]}}, data_out2_18} << 5);
assign LV00[103*20 +: 20] = ~({{8{data_out2_19[11]}}, data_out2_19} << 1);
assign LV00[104*20 +: 20] = ~({{8{data_out2_19[11]}}, data_out2_19} << 2);
assign LV00[105*20 +: 20] = ~({{8{data_out2_19[11]}}, data_out2_19} << 3);
assign LV00[106*20 +: 20] = ({{8{data_out2_20[11]}}, data_out2_20} << 1);
assign LV00[107*20 +: 20] = ({{8{data_out2_21[11]}}, data_out2_21} << 1);
assign LV00[108*20 +: 20] = ({{8{data_out2_21[11]}}, data_out2_21} << 5);
assign LV00[109*20 +: 20] = {{8{data_out2_22[11]}}, data_out2_22};
assign LV00[110*20 +: 20] = ({{8{data_out2_22[11]}}, data_out2_22} << 1);
assign LV00[111*20 +: 20] = ({{8{data_out2_22[11]}}, data_out2_22} << 2);
assign LV00[112*20 +: 20] = ({{8{data_out2_22[11]}}, data_out2_22} << 3);
assign LV00[113*20 +: 20] = ({{8{data_out2_22[11]}}, data_out2_22} << 4);
assign LV00[114*20 +: 20] = ({{8{data_out2_23[11]}}, data_out2_23} << 3);
assign LV00[115*20 +: 20] = ({{8{data_out2_24[11]}}, data_out2_24} << 1);
assign LV00[116*20 +: 20] = ({{8{data_out2_24[11]}}, data_out2_24} << 2);
assign LV00[117*20 +: 20] = ~{{8{data_out3_0[11]}}, data_out3_0};
assign LV00[118*20 +: 20] = ~({{8{data_out3_0[11]}}, data_out3_0} << 2);
assign LV00[119*20 +: 20] = ~({{8{data_out3_0[11]}}, data_out3_0} << 4);
assign LV00[120*20 +: 20] = ~{{8{data_out3_1[11]}}, data_out3_1};
assign LV00[121*20 +: 20] = ~({{8{data_out3_1[11]}}, data_out3_1} << 3);
assign LV00[122*20 +: 20] = ~({{8{data_out3_1[11]}}, data_out3_1} << 4);
assign LV00[123*20 +: 20] = ~({{8{data_out3_2[11]}}, data_out3_2} << 2);
assign LV00[124*20 +: 20] = ~({{8{data_out3_2[11]}}, data_out3_2} << 3);
assign LV00[125*20 +: 20] = ~({{8{data_out3_2[11]}}, data_out3_2} << 4);
assign LV00[126*20 +: 20] = ~{{8{data_out3_3[11]}}, data_out3_3};
assign LV00[127*20 +: 20] = ~({{8{data_out3_3[11]}}, data_out3_3} << 1);
assign LV00[128*20 +: 20] = ~({{8{data_out3_3[11]}}, data_out3_3} << 5);
assign LV00[129*20 +: 20] = {{8{data_out3_4[11]}}, data_out3_4};
assign LV00[130*20 +: 20] = ({{8{data_out3_4[11]}}, data_out3_4} << 1);
assign LV00[131*20 +: 20] = ({{8{data_out3_4[11]}}, data_out3_4} << 2);
assign LV00[132*20 +: 20] = ({{8{data_out3_4[11]}}, data_out3_4} << 3);
assign LV00[133*20 +: 20] = ~({{8{data_out3_5[11]}}, data_out3_5} << 2);
assign LV00[134*20 +: 20] = ~({{8{data_out3_5[11]}}, data_out3_5} << 4);
assign LV00[135*20 +: 20] = ~{{8{data_out3_6[11]}}, data_out3_6};
assign LV00[136*20 +: 20] = ~({{8{data_out3_6[11]}}, data_out3_6} << 5);
assign LV00[137*20 +: 20] = {{8{data_out3_7[11]}}, data_out3_7};
assign LV00[138*20 +: 20] = ({{8{data_out3_7[11]}}, data_out3_7} << 2);
assign LV00[139*20 +: 20] = {{8{data_out3_8[11]}}, data_out3_8};
assign LV00[140*20 +: 20] = ({{8{data_out3_8[11]}}, data_out3_8} << 4);
assign LV00[141*20 +: 20] = ~({{8{data_out3_9[11]}}, data_out3_9} << 1);
assign LV00[142*20 +: 20] = ~{{8{data_out3_10[11]}}, data_out3_10};
assign LV00[143*20 +: 20] = ~({{8{data_out3_10[11]}}, data_out3_10} << 1);
assign LV00[144*20 +: 20] = ~({{8{data_out3_10[11]}}, data_out3_10} << 2);
assign LV00[145*20 +: 20] = ~{{8{data_out3_11[11]}}, data_out3_11};
assign LV00[146*20 +: 20] = ~({{8{data_out3_11[11]}}, data_out3_11} << 2);
assign LV00[147*20 +: 20] = ~({{8{data_out3_11[11]}}, data_out3_11} << 3);
assign LV00[148*20 +: 20] = ({{8{data_out3_12[11]}}, data_out3_12} << 1);
assign LV00[149*20 +: 20] = ({{8{data_out3_13[11]}}, data_out3_13} << 2);
assign LV00[150*20 +: 20] = ~({{8{data_out3_14[11]}}, data_out3_14} << 5);
assign LV00[151*20 +: 20] = ~{{8{data_out3_15[11]}}, data_out3_15};
assign LV00[152*20 +: 20] = ~({{8{data_out3_17[11]}}, data_out3_17} << 1);
assign LV00[153*20 +: 20] = ~({{8{data_out3_17[11]}}, data_out3_17} << 4);
assign LV00[154*20 +: 20] = ~({{8{data_out3_18[11]}}, data_out3_18} << 1);
assign LV00[155*20 +: 20] = ~({{8{data_out3_18[11]}}, data_out3_18} << 2);
assign LV00[156*20 +: 20] = ~({{8{data_out3_18[11]}}, data_out3_18} << 4);
assign LV00[157*20 +: 20] = ~{{8{data_out3_19[11]}}, data_out3_19};
assign LV00[158*20 +: 20] = ~({{8{data_out3_19[11]}}, data_out3_19} << 2);
assign LV00[159*20 +: 20] = ~{{8{data_out3_20[11]}}, data_out3_20};
assign LV00[160*20 +: 20] = ~({{8{data_out3_20[11]}}, data_out3_20} << 1);
assign LV00[161*20 +: 20] = ~({{8{data_out3_20[11]}}, data_out3_20} << 3);
assign LV00[162*20 +: 20] = ~({{8{data_out3_21[11]}}, data_out3_21} << 4);
assign LV00[163*20 +: 20] = ~{{8{data_out3_22[11]}}, data_out3_22};
assign LV00[164*20 +: 20] = ~({{8{data_out3_22[11]}}, data_out3_22} << 1);
assign LV00[165*20 +: 20] = ~({{8{data_out3_22[11]}}, data_out3_22} << 4);
assign LV00[166*20 +: 20] = {{8{data_out3_23[11]}}, data_out3_23};
assign LV00[167*20 +: 20] = ({{8{data_out3_23[11]}}, data_out3_23} << 4);
assign LV00[168*20 +: 20] = ({{8{data_out3_24[11]}}, data_out3_24} << 2);
assign LV00[169*20 +: 20] = {{1{EXP_BIAS[11]}}, EXP_BIAS, 7'd0} + 20'd87;

// 3:2 compression : 170 -> 114 -> 76 -> 51 -> 34 -> 23 -> 16 -> 11 -> 8 -> 6 -> 4 -> 3 -> 2
// Pipeline cut after CSA07. CSA08+ and Kogge-Stone use registered MID07.
reg [11*20-1:0] MID07;
genvar cg;
generate
  wire [114*20-1:0] LV01;
  for (cg = 0; cg < 56; cg = cg + 1) begin : CSA01
    wire [19:0] a = LV00[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV00[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV00[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV01[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV01[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV01[112*20 +: 20] = LV00[168*20 +: 20];
  assign LV01[113*20 +: 20] = LV00[169*20 +: 20];
  wire [76*20-1:0] LV02;
  for (cg = 0; cg < 38; cg = cg + 1) begin : CSA02
    wire [19:0] a = LV01[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV01[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV01[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV02[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV02[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [51*20-1:0] LV03;
  for (cg = 0; cg < 25; cg = cg + 1) begin : CSA03
    wire [19:0] a = LV02[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV02[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV02[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV03[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV03[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV03[50*20 +: 20] = LV02[75*20 +: 20];
  wire [34*20-1:0] LV04;
  for (cg = 0; cg < 17; cg = cg + 1) begin : CSA04
    wire [19:0] a = LV03[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV03[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV03[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV04[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV04[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [23*20-1:0] LV05;
  for (cg = 0; cg < 11; cg = cg + 1) begin : CSA05
    wire [19:0] a = LV04[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV04[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV04[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV05[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV05[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV05[22*20 +: 20] = LV04[33*20 +: 20];
  wire [16*20-1:0] LV06;
  for (cg = 0; cg < 7; cg = cg + 1) begin : CSA06
    wire [19:0] a = LV05[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV05[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV05[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV06[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV06[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV06[14*20 +: 20] = LV05[21*20 +: 20];
  assign LV06[15*20 +: 20] = LV05[22*20 +: 20];
  wire [11*20-1:0] LV07;
  for (cg = 0; cg < 5; cg = cg + 1) begin : CSA07
    wire [19:0] a = LV06[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV06[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV06[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV07[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV07[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV07[10*20 +: 20] = LV06[15*20 +: 20];
  wire [8*20-1:0] LV08;
  for (cg = 0; cg < 3; cg = cg + 1) begin : CSA08
    wire [19:0] a = MID07[(3*cg+0)*20 +: 20];
    wire [19:0] b = MID07[(3*cg+1)*20 +: 20];
    wire [19:0] c = MID07[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV08[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV08[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV08[6*20 +: 20] = MID07[9*20 +: 20];
  assign LV08[7*20 +: 20] = MID07[10*20 +: 20];
  wire [6*20-1:0] LV09;
  for (cg = 0; cg < 2; cg = cg + 1) begin : CSA09
    wire [19:0] a = LV08[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV08[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV08[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV09[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV09[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV09[4*20 +: 20] = LV08[6*20 +: 20];
  assign LV09[5*20 +: 20] = LV08[7*20 +: 20];
  wire [4*20-1:0] LV10;
  for (cg = 0; cg < 2; cg = cg + 1) begin : CSA10
    wire [19:0] a = LV09[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV09[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV09[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV10[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV10[(2*cg+1)*20 +: 20] = cy << 1;
  end
  wire [3*20-1:0] LV11;
  for (cg = 0; cg < 1; cg = cg + 1) begin : CSA11
    wire [19:0] a = LV10[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV10[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV10[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV11[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV11[(2*cg+1)*20 +: 20] = cy << 1;
  end
  assign LV11[2*20 +: 20] = LV10[3*20 +: 20];
  wire [2*20-1:0] LV12;
  for (cg = 0; cg < 1; cg = cg + 1) begin : CSA12
    wire [19:0] a = LV11[(3*cg+0)*20 +: 20];
    wire [19:0] b = LV11[(3*cg+1)*20 +: 20];
    wire [19:0] c = LV11[(3*cg+2)*20 +: 20];
    wire [19:0] cy = (a & b) | (b & c) | (a & c);
    assign LV12[(2*cg+0)*20 +: 20] = a ^ b ^ c;
    assign LV12[(2*cg+1)*20 +: 20] = cy << 1;
  end
endgenerate

reg vc_s1;

// Stage 2: remaining CSA (combo from MID07) plus the one Kogge-Stone add.
wire [19:0] ksg [0:5];
wire [19:0] ksp [0:5];
assign ksg[0] = LV12[0*20 +: 20] & LV12[1*20 +: 20];
assign ksp[0] = LV12[0*20 +: 20] ^ LV12[1*20 +: 20];
genvar kl, ki;
generate
  for (kl = 1; kl <= 5; kl = kl + 1) begin : KSL
    for (ki = 0; ki < 20; ki = ki + 1) begin : KSB
      if (ki >= (1 << (kl-1))) begin : UPD
        assign ksg[kl][ki] = ksg[kl-1][ki] | (ksp[kl-1][ki] & ksg[kl-1][ki-(1<<(kl-1))]);
        assign ksp[kl][ki] = ksp[kl-1][ki] & ksp[kl-1][ki-(1<<(kl-1))];
      end else begin : KEEP
        assign ksg[kl][ki] = ksg[kl-1][ki];
        assign ksp[kl][ki] = ksp[kl-1][ki];
      end
    end
  end
endgenerate
wire [19:0] ks_sum;
assign ks_sum[0] = ksp[0][0];
generate
  for (ki = 1; ki < 20; ki = ki + 1) begin : KSS
    assign ks_sum[ki] = ksp[0][ki] ^ ksg[5][ki-1];
  end
endgenerate

always @ (posedge clk) begin
	if(~rst_n) begin
		vc_s1 <= 0;
		valid_out_calc <= 0;
		conv_out_calc <= 0;
	end
	else begin
		MID07 <= LV07;
		// Toggling Valid Output Signal (unchanged sequence, on the stage-1 flag)
		if(valid_out_buf == 1) begin
			if(vc_s1 == 1)
				vc_s1 <= 0;
			else
				vc_s1 <= 1;
		end

		// stage 2 : resolve the carries and slice
		conv_out_calc  <= ks_sum[19:6];
		valid_out_calc <= vc_s1;
	end
end

endmodule



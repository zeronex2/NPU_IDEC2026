/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : top_tb.v
 *  Written by : Kang, Bo Young
 *  Written on : Oct 15, 2021
 *  Version    : 21.2
 *  Design     : Testbench for CNN MNIST dataset - single input image
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: top_tb
 *------------------------------------------------------------------*/
`timescale 1ps/1ps
module top_tb();
parameter DATA_BITS = 8;
reg clk, rst_n;
reg [7:0] pixels [0:783];
reg [9:0] img_idx;
reg [7:0] data_in;

wire [3:0] decision;
wire valid_out_6;






// Clock generation
always #5 clk = ~clk;

// Read image text file
initial begin
  $readmemh("D:/npu_final/Reference code/Reference code/Reference code/cnn_verilog/data/2_0.txt", pixels);
  clk <= 1'b0;
  rst_n <= 1'b1;
  #3
  rst_n <= 1'b0;
  #3
  rst_n <= 1'b1;
end
 
always @(posedge clk) begin
  if(~rst_n) begin
    img_idx <= 0;
  end else begin
    if(img_idx < 10'd784) begin
      img_idx <= img_idx + 1'b1;
    end
    data_in <= pixels[img_idx];
  end
end
always @(*) begin
    if(valid_out_6==1) #5 $finish;
end
chip chip1 (
clk,rst_n,data_in,decision,valid_out_6);






endmodule

/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : top_tb_2.v
 *  Written by : Kang, Bo Young
 *  Written on : Oct 20, 2021
 *  Version    : 21.2
 *  Design     : Testbench for CNN MNIST dataset - multiple input image (1000)
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: top_tb_2
 *------------------------------------------------------------------*/
/*------------------------------------------------------------------------
 *
 *  Copyright (c) 2021 by Bo Young Kang, All rights reserved.
 *
 *  File name  : top_tb.v
 *  Written by : Kang, Bo Young
 *  Written on : Oct 15, 2021
 *  Version    : 21.2
 *  Design     : Testbench for CNN MNIST dataset - single input image
 *
 *------------------------------------------------------------------------*/

/*-------------------------------------------------------------------
 *  Module: top_tb
 *------------------------------------------------------------------*/
`timescale 1ps/1ps


module top_tb_1000();

reg clk, rst_n;
reg [7:0] pixels [0:783999];
reg [9:0] img_idx;
reg [7:0] data_in;
reg [9:0] cnt;  // num of input image (1000)
reg [9:0] input_cnt;
reg [9:0] rand_num; // 1000
reg state;

integer i_cnt;  // loop variable
reg [9:0] accuracy; // hit/miss count (1000)


wire [3:0] decision;
wire valid_out_6;






// Clock generation
always #5 clk = ~clk;

// Read image text file

 

chip chip1 (
clk,rst_n,data_in,decision,valid_out_6);




// Read image text file
initial begin
  $readmemh("D:/npu_final/Reference code/Reference code/Reference code/cnn_verilog/data/input_1000.txt", pixels);
  cnt <= 0;
  img_idx <= 0;
  clk <= 1'b0;
  input_cnt <= -1;
  rst_n <= 1'b1;
  rand_num <= 1'b0;
  accuracy <= 0;

  #3
  rst_n <= 1'b0;
  
  #3
  rst_n <= 1'b1;
end

always @(posedge clk) begin
  if(~rst_n) begin

    #3
    rst_n <= 1'b1;
  end else begin
    // decision done
    if(valid_out_6 == 1'b1) begin
      if(state !== 1'bx) begin
        if(cnt % 10 == 1) begin
          if(rand_num % 10 == decision) begin
            $display("%0dst input image : original value = %0d, decision = %0d at %0t ps ==> Success", cnt, rand_num % 10, decision, $time);
            accuracy <= accuracy + 1'b1;
          end else begin
            $display("%0dst input image : original value = %0d, decision = %0d at %0t ps ==> Fail", cnt, rand_num % 10, decision, $time);
          end
        end else if(cnt % 10 == 2) begin
          if(rand_num % 10 == decision) begin
            $display("%0dnd input image : original value = %0d, decision = %0d at %0t ps ==> Success", cnt, rand_num % 10, decision, $time);
            accuracy <= accuracy + 1'b1;
          end else begin
            $display("%0dnd input image : original value = %0d, decision = %0d at %0t ps ==> Fail", cnt, rand_num % 10, decision, $time);
          end
        end else if(cnt % 10 == 3)
          if(rand_num % 10 == decision) begin
            $display("%0drd input image : original value = %0d, decision = %0d at %0t ps ==> Success", cnt, rand_num % 10, decision, $time);
            accuracy <= accuracy + 1'b1;
          end else begin
            $display("%0drd input image : original value = %0d, decision = %0d at %0t ps ==> Fail", cnt, rand_num % 10, decision, $time);
          end
        else begin
          if(rand_num % 10 == decision) begin
            $display("%0dth input image : original value = %0d, decision = %0d at %0t ps ==> Success", cnt, rand_num % 10, decision, $time);
            accuracy <= accuracy + 1'b1;
          end else begin
            $display("%0dth input image : original value = %0d, decision = %0d at %0t ps ==> Fail", cnt, rand_num % 10, decision, $time);
          end
        end
      end

      state <= 1'b0;
      rst_n <= 1'b0;
      input_cnt <= input_cnt + 1'b1;
      rand_num <= $urandom_range(0, 1000);
      //rand_num <= rand_num + 1'b1;
    end

    if(state == 1'b0) begin
      //data_in <= pixels[cnt*784 + img_idx];
      data_in <= pixels[rand_num*784 + img_idx];
      //data_in <= pixel[img_idx];
      img_idx <= img_idx + 1'b1;

      if(img_idx == 10'd784) begin
        cnt <= cnt + 1'b1;

        if(cnt == 10'd1000) begin
          $display("\n\n------ Final Accuracy for 1000 Input Image ------");
          $display("Accuracy : %3d%%", accuracy/10);
          $stop;
        end
        img_idx <= 0;
        state <= 1'b1;  // done
      end
    end
  end
end

endmodule

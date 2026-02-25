/* Roy Vicerra (rvicerra)
 * CPE 442-01
 * Lab 5
*/


#include <iostream>
#include <opencv2/opencv.hpp>
#include <arm_neon.h>
#include "filters.hpp"


// Function: Convert OpenCV RGB Mat Image Frame to Grayscale Mat.
// Convert BGR (CV_8UC3) -> Grayscale (CV_8UC1) using ITU-R (BT.709) Algorithm
cv::Mat to442_grayscale(const cv::Mat &RGBFrame) {
  // Check if frame is an OpenCV RGB Mat (Close program if not)
  CV_Assert(RGBFrame.type() == CV_8UC3);
  
  // Allocate memory for Grayscale Mat Frame
  cv::Mat outFrame;
  outFrame.create(RGBFrame.rows, RGBFrame.cols, CV_8UC1);

  // Turn Grayscale Factors from floating point to integer (Factor by 256)
  const uint8_t kR = 0.2126 * INT_CONV_FACTOR;
  const uint8_t kG = 0.7152 * INT_CONV_FACTOR;
  const uint8_t kB = 0.0722 * INT_CONV_FACTOR;
  const uint8x8_t kR_vec = vdup_n_u8(kR);     // Create vector copy of static variable
  const uint8x8_t kG_vec = vdup_n_u8(kG);
  const uint8x8_t kB_vec = vdup_n_u8(kB);

  // Iterate through every pixel in every row
  for (int rowIDX = 0; rowIDX < RGBFrame.rows; rowIDX++) {
    // Create vector pointers for input pixel (3 Unsigned Bytes) and output pixel (1 Unsigned Byte)
    // Note: OpenCV uses uchar instead of uint8_t but are functionally equivalent
    const cv::Vec3b* inPixel = RGBFrame.ptr<cv::Vec3b>(rowIDX);
    const uint8_t* inPixel_1b = (const uint8_t*) inPixel;   // Separate RGB Frame by Bytes instead of Pixels
    uchar* outPixel = outFrame.ptr<uchar>(rowIDX);

    // Iterate every 8 Pixels (columns) in row
    int colIDX;
    for (colIDX = 0; colIDX <= (RGBFrame.cols - 8); colIDX += 8) {
      // Obtain RGB values by grouping 8 RGB Pixels into {B, G, R} vectors
      // ie) {B0 G0 R0}, {B1 G1 R1}, {B2 G2 R2} ... {B7 G7 R7}
      //     0 Vec3b      1 Vec3b     2 Vec3b         7 Vec3b
      uint8x8x3_t BGR = vld3_u8(inPixel_1b + (3 * colIDX));  // Iterate every 24 bytes
      uint8x8_t B = BGR.val[0];
      uint8x8_t G = BGR.val[1];
      uint8x8_t R = BGR.val[2];

      // Calculate Grayscale Conversion (ITU-R (BT.709))
      // Temporarily convert to 16-bit value to account for integer multiplication
      uint16x8_t acc = vmull_u8(R, kR_vec);
      acc = vmlal_u8(acc, G, kG_vec);
      acc = vmlal_u8(acc, B, kB_vec);

      // Divide by 256 to get true ITU-R Values, then reduce back to 8-bit value
      uint8x8_t gray = vrshrn_n_u16(acc, INT_CONV_BITS);

      // Store grayscale value to Output Frame
      vst1_u8(outPixel + colIDX, gray);
    }

    // Account for Remaining Pixels at the end of row (Apply Filter Per Pixel)
    if (colIDX < RGBFrame.cols) {
      for (colIDX -= 8; colIDX < RGBFrame.cols; colIDX++) {
        // Separate Each Byte into B, G, R pixels respectively
        uint8_t B = inPixel[colIDX][0];
        uint8_t G = inPixel[colIDX][1];
        uint8_t R = inPixel[colIDX][2];

        // Apply ITU-R Conversion (in integer form, Factor by 256)
        int gray = (kR * (int) R) + (kG * (int) G) + (kB * (int) B);
        // Divide by 256 and Round UP
        gray = (gray + 128) >> 8;

        // Store Grayscale Pixel to Output Frame as unsigned byte
        outPixel[colIDX] = (uchar) gray;
      }
    }
  }

  // Return Output Mat Frame
  return outFrame;
}


// Function: Convert Grayscale Mat to Sobel Filter Applied Mat.
// Convert Grayscale (CV_8UC1) -> Sobel (CV_8UC1)
// Gx = | -1  0  1  |        Gy = |  1  2  1  |
//      | -2  0  2  |             |  0  0  0  |
//      | -1  0  1  |             | -1 -2 -1  |
//
// |G| = |Gx| + |Gy|
//
cv::Mat to442_sobel(const cv::Mat &grayFrame) {
  // Check if frame is an OpenCV Grayscale MAT (Close program if not)
  CV_Assert(grayFrame.type() == CV_8UC1);

  // Allocate memory for Sobel Mat Frame (Calloc)
  cv::Mat outFrame;
  outFrame.create(grayFrame.rows, grayFrame.cols, CV_8UC1);
  outFrame.setTo(cv::Scalar(0));

  // Calculate the Convolution of Sobel Kernels with 3x3 Pixel Frame
  // Iterate through every pixel in every row EXCEPT border
  for (int rowIDX = 1; rowIDX < (grayFrame.rows - 1); rowIDX++) {
    // Create 1 Unsigned Byte pointers for Rows of Previous Pixel, Current Pixel, and Next Pixel, and Output Pixel
    const uchar* prev_pixel_row = grayFrame.ptr<uchar>(rowIDX - 1);
    const uchar* curr_pixel_row = grayFrame.ptr<uchar>(rowIDX);
    const uchar* next_pixel_row = grayFrame.ptr<uchar>(rowIDX + 1);
    uchar* outPixel_row = outFrame.ptr<uchar>(rowIDX);

    // Iterate through every 8 Pixels (columns) EXCEPT border
    int colIDX;
    for (colIDX = 1; colIDX <= (grayFrame.cols - BORDER_LEN - 8); colIDX += 8) {
      const uint8_t* pixelPtr_prevRow = prev_pixel_row + (colIDX - 1);
      const uint8_t* pixelPtr_currRow = curr_pixel_row + (colIDX - 1);
      const uint8_t* pixelPtr_nextRow = next_pixel_row + (colIDX - 1);

      // Grab 8 Pixels for Left, Middle, and Right Columns Each
      uint8x8_t topLeft_pixels = vld1_u8(pixelPtr_prevRow);
      uint8x8_t topMid_pixels = vld1_u8(pixelPtr_prevRow + 1);
      uint8x8_t topRight_pixels = vld1_u8(pixelPtr_prevRow + 2);

      uint8x8_t midLeft_pixels = vld1_u8(pixelPtr_currRow);
      // uint8x8_t midMid_pixels = vld1_u8(pixelPtr_currRow + 1);   // Removed due to redundancy
      uint8x8_t midRight_pixels = vld1_u8(pixelPtr_currRow + 2);

      uint8x8_t botLeft_pixels = vld1_u8(pixelPtr_nextRow);
      uint8x8_t botMid_pixels = vld1_u8(pixelPtr_nextRow + 1);
      uint8x8_t botRight_pixels = vld1_u8(pixelPtr_nextRow + 2);

      // Gx = (topRight - topLeft) + 2*(midRight - midLeft) + (botRight - botLeft)
      //              Gx_top                Gx_mid                  Gx_bot
      int16x8_t Gx_top = (int16x8_t) vsubl_u8(topRight_pixels, topLeft_pixels);
      int16x8_t Gx_mid = vshlq_n_s16((int16x8_t) vsubl_u8(midRight_pixels, midLeft_pixels), 1);  // Shift by one bit (multiply by 2)
      int16x8_t Gx_bot = (int16x8_t) vsubl_u8(botRight_pixels, botLeft_pixels);
      int16x8_t Gx = vaddq_s16(Gx_top, Gx_mid);
      Gx = vaddq_s16(Gx, Gx_bot);

      // Gy = (topLeft + 2*topMid + topRight) - (botLeft + 2*botMid + botRight)
      //          Gy_top (Positive Only)             Gy_bot (Positive Only)
      uint16x8_t Gy_top = vaddl_u8(topLeft_pixels, topRight_pixels);
      Gy_top = vaddq_u16(Gy_top, vshll_n_u8(topMid_pixels, 1));
      uint16x8_t Gy_bot = vaddl_u8(botLeft_pixels, botRight_pixels);
      Gy_bot = vaddq_u16(Gy_bot, vshll_n_u8(botMid_pixels, 1));
      int16x8_t Gy = (int16x8_t) vsubq_u16(Gy_top, Gy_bot);

      // G = |Gx| + |Gy|
      uint16x8_t G = vaddq_u16( (uint16x8_t) vabsq_s16(Gx), (uint16x8_t) vabsq_s16(Gy));
      // Restrict to Byte size
      uint8x8_t G_byte8 = vqmovn_u16(G);
      
      // Store Sobel Value to Output Frame
      vst1_u8(outPixel_row + colIDX, G_byte8);
    }


    // Handle remaining pixels individually
    if (colIDX < grayFrame.cols) {
      for (colIDX -= 8; colIDX < (grayFrame.cols - 1); colIDX++) {
        // Calculate Gx Kernel
        int Gx = 0;

        // First Row of Kernel
        Gx += -1 * prev_pixel_row[colIDX - 1];
        //Gx += 0 * prev_pixel_row[colIDX];       // Removed due to redundancy
        Gx += 1 * prev_pixel_row[colIDX + 1];
  
        // Second Row of Kernel
        Gx += -2 * curr_pixel_row[colIDX - 1];
        //Gx += 0 * curr_pixel_row[colIDX];
        Gx += 2 * curr_pixel_row[colIDX + 1];

        // Third Row of Kernel
        Gx += -1 * next_pixel_row[colIDX - 1];
        //Gx += 0 * next_pixel_row[colIDX];
        Gx += 1 * next_pixel_row[colIDX + 1];

        // Calculate Gy Kernel
        int Gy = 0;

        // First Row of Kernel
        Gy += 1 * prev_pixel_row[colIDX - 1];
        Gy += 2 * prev_pixel_row[colIDX];
        Gy += 1 * prev_pixel_row[colIDX + 1];

        // Second Row of Kernel
        //Gy += 0 * curr_pixel_row[colIDX - 1];
        //Gy += 0 * curr_pixel_row[colIDX];
        //Gy += 0 * curr_pixel_row[colIDX + 1];

        // Third Row of Kernel
        Gy += -1 * next_pixel_row[colIDX - 1];
        Gy += -2 * next_pixel_row[colIDX];
        Gy += -1 * next_pixel_row[colIDX + 1];

      
        // Calculate Edge Strength, |G|
        int G = abs(Gx) + abs(Gy);
        // Restrict to size 1 Unsigned Byte (0 - 255)
        if (G > 255) {
          G = 255;
        } 

        // Output Edge Strength to Output Pixel
        outPixel_row[colIDX] = (uchar) G;
      }
    }
  }
  
  // Return Output Mat Sobel Frame
  return outFrame;
}


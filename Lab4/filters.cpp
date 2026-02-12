/* Roy Vicerra (rvicerra)
 * CPE 442-01
 * Lab 3
*/


#include <iostream>
#include <opencv2/opencv.hpp>
#include "filters.hpp"


// Function: Convert OpenCV RGB Mat Image Frame to Grayscale Mat.
// Convert BGR (CV_8UC3) -> Grayscale (CV_8UC1) using ITU-R (BT.709) Algorithm
cv::Mat to442_grayscale(const cv::Mat &RGBFrame) {
  // Check if frame is an OpenCV RGB Mat (Close program if not)
  CV_Assert(RGBFrame.type() == CV_8UC3);
  
  // Allocate memory for Grayscale Mat Frame
  cv::Mat outFrame;
  outFrame.create(RGBFrame.rows, RGBFrame.cols, CV_8UC1);

  // Iterate through every pixel in every row
  for (int rowIDX = 0; rowIDX < RGBFrame.rows; rowIDX++) {
    // Create vector pointers for input pixel (3 Unsigned Bytes) and output pixel (1 Unsigned Byte)
    // Note: OpenCV uses uchar instead of uint8_t but are functionally equivalent
    const cv::Vec3b* inPixel = RGBFrame.ptr<cv::Vec3b>(rowIDX);
    uchar* outPixel = outFrame.ptr<uchar>(rowIDX);

    // Iterate through every column in row
    for (int colIDX = 0; colIDX < RGBFrame.cols; colIDX++) {
      // Obtain RGB values of each pixel and calculate Grayscale pixel
      int B = inPixel[colIDX][0];
      int G = inPixel[colIDX][1];
      int R = inPixel[colIDX][2];

      // Calculate Grayscale Conversion (ITU-R (BT.709))
      int gray = (0.2126 * R) + (0.7152 * G) + (0.0722 * B);
      // Constrain Grayscale value to one byte (0-255)
      if (gray < 0) {
        gray = 0;

      } else if (gray > 255) {
        gray = 255;

      }

      // Assign Grascale value to output pixel
      outPixel[colIDX] = (uchar) gray;
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

    // Iterate through every pixel in every column EXCEPT border
    for (int colIDX = 1; colIDX < (grayFrame.cols - 1); colIDX++) {
      // Calculate Gx Kernel
      int Gx = 0;

      // First Row of Kernel
      Gx += -1 * prev_pixel_row[colIDX - 1];
      Gx += 0 * prev_pixel_row[colIDX];
      Gx += 1 * prev_pixel_row[colIDX + 1];
  
      // Second Row of Kernel
      Gx += -2 * curr_pixel_row[colIDX - 1];
      Gx += 0 * curr_pixel_row[colIDX];
      Gx += 2 * curr_pixel_row[colIDX + 1];

      // Third Row of Kernel
      Gx += -1 * next_pixel_row[colIDX - 1];
      Gx += 0 * next_pixel_row[colIDX];
      Gx += 1 * next_pixel_row[colIDX + 1];

      // Calculate Gy Kernel
      int Gy = 0;

      // First Row of Kernel
      Gy += 1 * prev_pixel_row[colIDX - 1];
      Gy += 2 * prev_pixel_row[colIDX];
      Gy += 1 * prev_pixel_row[colIDX + 1];

      // Second Row of Kernel
      Gy += 0 * curr_pixel_row[colIDX - 1];
      Gy += 0 * curr_pixel_row[colIDX];
      Gy += 0 * curr_pixel_row[colIDX + 1];

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
  
  // Return Output Mat Sobel Frame
  return outFrame;
}


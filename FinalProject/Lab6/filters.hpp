/* Roy Vicerra (rvicerra)
 * CPE 442-01
 * Lab 3
*/

#ifndef FILTERS_HPP
#define FILTERS_HPP

#include <iostream>

// MACROS //
#define INT_CONV_FACTOR 256
#define INT_CONV_BITS 8
#define BORDER_LEN 2


// Function: Convert OpenCV RGB Mat Image Frame to Grayscale Mat.
cv::Mat to442_grayscale(const cv::Mat &inFrame);

// Function: Convert Grayscale Mat to Sobel Filter Applied Mat.
cv::Mat to442_sobel(const cv::Mat &grayFrame);

#endif

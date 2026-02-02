/* 
 * Roy Vicerra (rvicerra)
 * CPE 442-01
 * Lab 3
*/


#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "filters.hpp"


// Function: Pull in frames from a video file/webcam one after another,
// apply grayscale and Sobel to the frame and display it to the user
// until video is complete or user stops the program.
int main(int argc, char** argv) {
  // Parse user input (Video Filename / Webcam [No Input])
  std::string filename;
  cv::VideoCapture vid;
  
  if (argc == 2) {
    filename = argv[1];
    vid.open(filename);
    if (!vid.isOpened()) {
      std::cerr << "ERROR: Couldn't open video file: " << filename << std::endl;

      return -1;
    }
    
  } else if (argc == 1) {
    vid.open(0);
    if (!vid.isOpened()) {
      std::cerr << "ERROR: Couldn't open webcam." << std::endl;

      return -1;
    }

  } else {
    std::cout << "Usage: " << argv[0] << " [video_filename] ( or leave empty to use webcam )." << std::endl;
 
    return -1;
  }

  // Alert user to end program when desired
  std::cout << "Press 'q' or ESC to quit." << std::endl;

  // Create display windows
  cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Grayscale", cv::WINDOW_AUTOSIZE);

// -------------------------------------------------- //

  // Handle Mat objects
  cv::Mat inFrame, outFrame;

  // Set Playback Framerate of Video
  double fps = vid.get(cv::CAP_PROP_FPS);
  int delay = 1;
  if (fps > 1.0) {
    delay = std::max(1, static_cast<int>(1000.0 / fps));
  }

  // Iterate through every frame in video
  while (1) {
    // Exit loop when end of frame has been reached (or error)
    if (!vid.read(inFrame)) {
      break;
    }

    // Display Original Frame to user window
    cv::imshow("Original", inFrame);

    // Convert RGB Frame to Grayscale Frame
    outFrame = to442_grayscale(inFrame);
    // Display Frame to user window
    cv::imshow("Grayscale", outFrame);

    // Convert Grayscale Frame to Sobel Frame
    outFrame = to442_sobel(outFrame);
    // Display Frame to user window
    cv::imshow("Sobel", outFrame);
  
    // Wait for User to Close Window
    int key = cv::waitKey(delay) & 0xFF;
    if (key == 27 || key == 'q') {
      break;
    }
  }

  // Close OpenCV Applications
  vid.release();
  cv::destroyAllWindows();


  return 0;
}


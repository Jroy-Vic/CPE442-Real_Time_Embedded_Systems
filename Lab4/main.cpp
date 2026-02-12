/*
 * Roy Vicerra (rvicerra)
 * CPE 442-01
 * Lab 4
*/


#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include "filters.hpp"


// Thread Prototypes:
void* inputParse_Thread1(void* vid_ptr);
void* processSegment_Thread2(void* seg_idx_ptr);
void* recombFrame_Thread3(void*);


// ------------------------------------------------------- //
// Struct Definitions //

// Struct Function: Monitor used to Handle Synchronization, Memory Sharing, and Handshake 
// between Thread1 and its children threads, Thread2.
struct monitor_t {
  // Create Global Mutex Lock, Conditional Variable for Parent Thread 
  // Conditional Variables for Each Child Thread, and Global Children Conditional Variable
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t child_thread_cond  = PTHREAD_COND_INITIALIZER;
  pthread_cond_t parent_thread_cond = PTHREAD_COND_INITIALIZER;
  pthread_cond_t output_thread_cond = PTHREAD_COND_INITIALIZER;

  // Global Data Accumulated from Parent Thread and Child Threads
  cv::Mat seg_in[4];          // Raw Frame Segment
  cv::Mat seg_out[4];         // Sobel Frame Segment
  cv::Rect ref_segs[4];       // Segment Rectangle
  cv::Rect worker_segs[4];    // Segment Rectangle (with Padding)

  // Child Thread Conditions
  uint8_t is_ready[4] = {0x0, 0x0 ,0x0 ,0x0};
  uint8_t is_done[4] = {0x0, 0x0, 0x0, 0x0};
  int frame_id = 0;
  uint8_t eof = 0x0;

  // Output Thread Conditions
  uint8_t output_ready = 0x0;
};


// Struct Function: Handles Child Worker Thread Arguments.
struct child_arg_t {
  int idx;    // Indicates which segment child thread Handles
};


// ------------------------------------------------------- //
// Global Variables //

// Create Monitor Variable for Thread1 and Thread2
monitor_t monitor;


// ------------------------------------------------------- //
// Thread Definitions //

// Thread Function: Pull in frames from a video file/webcam one after another,
// splitting each frame into four equal rows before moving on to the next;
// apply padding pixels when necessary and pass each segment to child threads.
void* inputParse_Thread1(void* vid_ptr) {
  // Recast vid_ptr (void*) into a VideoCapture object pointer
  cv::VideoCapture* vid = (cv::VideoCapture*) vid_ptr;

  // Check if vid is working properly; otherwise, end thread (program)
  if (!vid || !vid->isOpened()) {
    std::cerr << "inputParse_Thread1: invalid VideoCapture\n";
    return nullptr;
  }


  // -------------------------------------------------- //


  // Create Four Child Threads to Handle Worker Segments Continuously
  pthread_t child_threads[4];
  void* child_thread_status[4];
  child_arg_t args[4];
  for (uint8_t i = 0; i < 4; i++) {
    args[i].idx = i;
    pthread_create(&child_threads[i], NULL, processSegment_Thread2, (void*) &args[i]);
  }


  // Iterate through new frames until EOF or error
  while (1) {
    cv::Mat inFrame;
  
    // Kill Parent Thread if User requests exit
    pthread_mutex_lock(&monitor.mutex);
    uint8_t user_exit = monitor.eof;
    pthread_mutex_unlock(&monitor.mutex);
    if (user_exit) {
      break;
    }
    
    // Break out of loop once there are no more frames to be read after updating Monitor
    if(!vid->read(inFrame)) {
      // Lock all other threads; Only run parent thread
      pthread_mutex_lock(&monitor.mutex);

      // Update EOF Condition
      monitor.eof = 0x1;

      // Signal to ALL Child Threads and Output Thread that Parent Thread is Complete (Terminate All Threads)
      pthread_cond_broadcast(&monitor.child_thread_cond);
      pthread_cond_signal(&monitor.output_thread_cond);
      
      // Allow all other threads to pass
      pthread_mutex_unlock(&monitor.mutex);
      break;
    }


    // Compute four Horizontal rows
    int height_total = inFrame.rows;
    int width_seg = inFrame.cols;
    int height_seg = height_total / 4; 
      
    // Create Reference Segments (Used for Recombination)
    cv::Rect ref_segs_local[4] = {
      // Rect(x_topleft, y1_topleft, width, height)
      // NOTE: Apply pixel padding to top and bottom of middle two segments,
      // bottom of top segment, and top of bottom segment.
      cv::Rect (0, 0, width_seg, height_seg),                                        // 0 - Top Seg
      cv::Rect (0, height_seg, width_seg, height_seg),                               // 1 - Middle Top Seg
      cv::Rect (0, (height_seg * 2), width_seg, height_seg),                         // 2 - Middle Bottom Seg
      cv::Rect (0, (height_seg * 3), width_seg, (height_total - (height_seg * 3)))   // 3 - Bottom Seg
    };

    // Create Worker Segments (Used for Processing Thread)
    cv::Rect worker_segs_local[4] = {
      // Rect(x_topleft, y1_topleft, width, height)
      // NOTE: Apply pixel padding to top and bottom of middle two segments,
      // bottom of top segment, and top of bottom segment.
      cv::Rect (0, 0, width_seg, (height_seg + 1)),                                              // 0 - Top Seg
      cv::Rect (0, (height_seg - 1), width_seg, (height_seg + 2)),                               // 1 - Middle Top Seg
      cv::Rect (0, ((height_seg * 2) - 1), width_seg, (height_seg + 2)),                         // 2 - Middle Bottom Seg
      cv::Rect (0, ((height_seg * 3) - 1), width_seg, (height_total - (height_seg * 3) + 1))     // 3 - Bottom Seg
    };


    // Lock all other threads; allow only parent thread to run
    pthread_mutex_lock(&monitor.mutex);

    // Increment Frame Count/ID to indicate new frame is being processed
    monitor.frame_id++;
    
    // Fill Child Thread Buffer Data with metadata
    for (uint8_t i = 0; i < 4; i++) {
      monitor.ref_segs[i] = ref_segs_local[i];
      monitor.worker_segs[i] = worker_segs_local[i];

      // Allocates new memory to each segment and stores globally
      monitor.seg_in[i] = inFrame(worker_segs_local[i]).clone();  // inFrame(worker_segs_local[i]) takes segment of inFrame sepcified by worker_segs_local[i]
                                                                  // clone() is needed to copy data to a new memory address from inFrame

      // Update Thread Conditions
      monitor.is_ready[i] = 0x1;    // Activate Thread2
      monitor.is_done[i]= 0x0;

    }

    // Signal ALL Child Threads to process new frame
    pthread_cond_broadcast(&monitor.child_thread_cond);


    // Reopen Mutex to Child Threads; Wait for all four child threads to finish frame
    while (!(monitor.is_done[0] && monitor.is_done[1] && monitor.is_done[2] && monitor.is_done[3])) {
      pthread_cond_wait(&monitor.parent_thread_cond, &monitor.mutex);
    }

    // Signal Output Thread that new frame is is_ready
    monitor.output_ready = 0x1;
    pthread_cond_signal(&monitor.output_thread_cond);

    // Unlock Parent Thread Mutex
    pthread_mutex_unlock(&monitor.mutex);
  }

  // Once EOF has been reached (or error occurred), join threads together to end process
  for (uint8_t i = 0; i < 4; i++) {
    pthread_join(child_threads[i], &child_thread_status[i]);
  }

  std::cout << "Parent Thread done.\n";
  return nullptr;
}


// Thread Function: Apply filters to individual frame segment, send processed
// segment to global child thread, then continue working on the next segment.
void* processSegment_Thread2(void* seg_idx_ptr) {
  // Recast argument into Child Argument Object Pointer
  child_arg_t* arg = (child_arg_t*) seg_idx_ptr;
  // Determine which segment to process
  int idx = arg->idx;

  // Initialize Last Frame Index
  int frame_id_local = -1;

  while (1) {
    // Give single Child Thread permission to process
    pthread_mutex_lock(&monitor.mutex);
    
    // Put Child Thread to sleep and give Parent Thread access until new frame is processed
    while (!monitor.eof && (!monitor.is_ready[idx] || monitor.frame_id == frame_id_local)) {
      pthread_cond_wait(&monitor.child_thread_cond, &monitor.mutex);      
    }

    // Kill Child Thread once EOF is reached
    if (monitor.eof) {
      pthread_mutex_unlock(&monitor.mutex);
      break;
    }

    // Localize individual segment data to Child Thread
    cv::Mat seg_in_local = monitor.seg_in[idx];
    frame_id_local = monitor.frame_id;

    // Clear Child Thread is Ready Flag
    monitor.is_ready[idx] = 0x0;

    // Unlock Child Thread; Allow other threads to process while this thread applies filters locally
    pthread_mutex_unlock(&monitor.mutex);


    // Process Segment Mat (Apply Filters)
    cv::Mat grayscale_local = to442_grayscale(seg_in_local);
    cv::Mat sobel_local = to442_sobel(grayscale_local);


    // Update Monitor Status to Reflect Done Child Thread
    pthread_mutex_lock(&monitor.mutex);
    monitor.seg_out[idx] = sobel_local;
    monitor.is_done[idx] = 0x1;
    // Signal Parent Thread that one Child Thread is complete
    pthread_cond_signal(&monitor.parent_thread_cond);
    pthread_mutex_unlock(&monitor.mutex);
  }

  std::cout << "Child Thread " << idx << " done.\n";
  return nullptr;
}


// Thread Function: Wait for all four frame segments to be processed and received,
// then stitch them all back into a single frame.
void* recombFrame_Thread3(void*) {
  // Create a Single Display Window
  cv::namedWindow("Sobel", cv::WINDOW_AUTOSIZE);

  while (1) {
    // Give access to Output Thread
    pthread_mutex_lock(&monitor.mutex);

    while ((!monitor.eof) && (!monitor.output_ready)) {
      // Wait for Parent Thread to receive confirmation of ALL Child Thread completion
      pthread_cond_wait(&monitor.output_thread_cond, &monitor.mutex);
    }

    // Kill Output Thread if EOF
    if (monitor.eof) {
      pthread_mutex_unlock(&monitor.mutex);
      break;
    }


    // -------------------------------------------------- //
    

    // Recover Full Frame Size from Reference Segments
    int full_width = monitor.ref_segs[0].width;
    int full_height = (monitor.ref_segs[3].y + monitor.ref_segs[3].height);

    // Allocate Memory for Output Frame
    cv::Mat outFrame(full_height, full_width, CV_8UC1);

    // Save local copy of segment data
    cv::Mat seg_out_local[4];
    cv::Rect ref_segs_local[4];
    cv::Rect worker_segs_local[4];
    for (uint8_t i = 0; i < 4; i++) {
      seg_out_local[i] = monitor.seg_out[i];
      ref_segs_local[i] = monitor.ref_segs[i];
      worker_segs_local[i] = monitor.worker_segs[i]; 
    }

    // Allow other threads to pass
    pthread_mutex_unlock(&monitor.mutex); 

    // Stitch Sobel Segments together (Remove Padding)
    for (uint8_t i = 0; i < 4; i++) {
      // topPadding = 0 for Top Segment (i = 0), topPadding = 1 otherwise
      // bottomPadding = 0 for Bottom Segment (i = 3), bottomPadding = 1 otherwise
      int topPadding = ref_segs_local[i].y - worker_segs_local[i].y;
      int bottomPadding = (worker_segs_local[i].y + worker_segs_local[i].height) - (ref_segs_local[i].y + ref_segs_local[i].height); 

      cv::Rect remove_padding(0, topPadding, seg_out_local[i].cols, seg_out_local[i].rows - topPadding - bottomPadding);

      seg_out_local[i](remove_padding).copyTo(outFrame(ref_segs_local[i]));
    }


    // -------------------------------------------------- //


    // Display the Output Frame
    cv::imshow("Sobel", outFrame);

    // Handle User input
    int key = cv::waitKey(1) & 0xFF;
    if ((key == 27) || (key == 'q')) {
      // Set EOF and wake ALL threads when User requests exit
      pthread_mutex_lock(&monitor.mutex);
      monitor.eof = 0x1;
      pthread_cond_broadcast(&monitor.child_thread_cond);
      pthread_cond_signal(&monitor.parent_thread_cond);
      pthread_mutex_unlock(&monitor.mutex);

      // End Output Thread
      break;
    }
  }

  std::cout << "Output Thread done.\n";
  return nullptr;
}


// ------------------------------------------------------- //
// Main Process //

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


// -------------------------------------------------- //


  // Set Playback Framerate of Video
  double fps = vid.get(cv::CAP_PROP_FPS);
  int delay = 1;
  if (fps > 1.0) {
    delay = std::max(1, static_cast<int>(1000.0 / fps));
  }

 // Create Main Parent Thread and Output Thread
  pthread_t parent_thread, output_thread;
  void* parent_thread_status, *output_thread_status;
  pthread_create(&parent_thread, NULL, inputParse_Thread1, &vid);
  pthread_create(&output_thread, NULL, recombFrame_Thread3, NULL);

  // Wait for Parent Thread and Output Thread to Finish
  pthread_join(parent_thread, &parent_thread_status);
  pthread_join(output_thread, &output_thread_status);

  // Close OpenCV Applications
  vid.release();
  cv::destroyAllWindows();


  return 0;
}


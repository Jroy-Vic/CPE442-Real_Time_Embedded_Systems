/*
 * Roy Vicerra (rvicerra)
 * CPE 442-01
 * Lab 5
*/


#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include "filters.hpp"

// MACROS:
#define SEG_CNT     4
#define BUFF_SIZE   10

// Thread Prototypes:
void* inputParse_Thread1(void* vid_ptr);
void* processSegment_Thread2(void* seg_idx_ptr);
void* recombFrame_Thread3(void*);


// ------------------------------------------------------- //
// Struct Definitions //

// Struct Function: Frame Handler used to Store Frame Metadata as Pipeline Buffer.
struct frame_handler_t {
  cv::Mat seg_in[SEG_CNT];          // Raw Frame Segment
  cv::Mat seg_out[SEG_CNT];         // Sobel Frame Segment
};


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
  frame_handler_t frame_InUse[BUFF_SIZE] = {};
  uint8_t frame_handler_state[BUFF_SIZE] = {0};   // States: 0 = Not Ready, 1 = Ready
  cv::Rect ref_segs[SEG_CNT];     // Segment Rectangle
  cv::Rect worker_segs[SEG_CNT];  // Segment Rectangle (with Padding)

  // Child Thread Conditions
  uint8_t parent_ready = 0x1;
  uint8_t eof = 0x0;

  // Output Thread Conditions
  uint8_t seg_done[BUFF_SIZE][SEG_CNT] = {{0}};
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
  pthread_t child_threads[SEG_CNT];
  void* child_thread_status[SEG_CNT];
  child_arg_t args[SEG_CNT];
  for (uint8_t i = 0; i < SEG_CNT; i++) {
    args[i].idx = i;
    pthread_create(&child_threads[i], NULL, processSegment_Thread2, (void*) &args[i]);
  }


  // Iterate through new frames until EOF or error
  while (1) {
    cv::Mat inFrame;
    static uint8_t user_exit = 0x0;
    static int frame_head = 0; 

    // Kill Parent Thread if User requests exit
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

    // -------------------------------------------------- //

    // Compute four Horizontal rows (Assume all the same after first frame)
    static uint8_t idx_lock = 0x0;
    if (!idx_lock) {
      idx_lock++;
      const int height_total = inFrame.rows;
      const int width_seg = inFrame.cols;
      const int height_seg = height_total / SEG_CNT; 
      
      // Create Reference Segments (Used for Recombination)
      const cv::Rect ref_segs_local[4] = {
        // Rect(x_topleft, y1_topleft, width, height)
        // NOTE: Apply pixel padding to top and bottom of middle two segments,
        // bottom of top segment, and top of bottom segment.
        cv::Rect (0, 0, width_seg, height_seg),                                        // 0 - Top Seg
        cv::Rect (0, height_seg, width_seg, height_seg),                               // 1 - Middle Top Seg
        cv::Rect (0, (height_seg * 2), width_seg, height_seg),                         // 2 - Middle Bottom Seg
       cv::Rect (0, (height_seg * 3), width_seg, (height_total - (height_seg * 3)))   // 3 - Bottom Seg
      };

      // Create Worker Segments (Used for Processing Thread)
      const cv::Rect worker_segs_local[4] = {
        // Rect(x_topleft, y1_topleft, width, height)
        // NOTE: Apply pixel padding to top and bottom of middle two segments,
        // bottom of top segment, and top of bottom segment.
        cv::Rect (0, 0, width_seg, (height_seg + 1)),                                              // 0 - Top Seg
        cv::Rect (0, (height_seg - 1), width_seg, (height_seg + 2)),                               // 1 - Middle Top Seg
        cv::Rect (0, ((height_seg * 2) - 1), width_seg, (height_seg + 2)),                         // 2 - Middle Bottom Seg
        cv::Rect (0, ((height_seg * 3) - 1), width_seg, (height_total - (height_seg * 3) + 1))     // 3 - Bottom Seg
      };

      // Save Globally
      pthread_mutex_lock(&monitor.mutex);
      for (uint8_t i = 0; i < SEG_CNT; i++) {
        monitor.ref_segs[i] = ref_segs_local[i];
        monitor.worker_segs[i] = worker_segs_local[i];
      }
      pthread_mutex_unlock(&monitor.mutex);
    }

    // -------------------------------------------------- //

    // Lock all other threads; allow only parent thread to run
    pthread_mutex_lock(&monitor.mutex);    

    // Split Frame only if there is space in the pipeline buffer
    while (monitor.frame_handler_state[frame_head] && !monitor.eof) {
      monitor.parent_ready = 0x0;
      pthread_cond_wait(&monitor.parent_thread_cond, &monitor.mutex);
    }
    monitor.parent_ready = 0x1;

    for (uint8_t i = 0; i < SEG_CNT; i++) {
      // Allocates new memory to each segment and stores globally
      monitor.frame_InUse[frame_head].seg_in[i] = inFrame(monitor.worker_segs[i]);  // inFrame(worker_segs_local[i]) takes segment of inFrame sepcified by worker_segs_local[i]
      
      // Clear Output Mats to be rewritten
      monitor.seg_done[frame_head][i] = 0x0;
      monitor.frame_InUse[frame_head].seg_out[i].release();
    }

    // Increment Frame Count/ID to indicate new frame is being processed, then signal Child Threads
    monitor.frame_handler_state[frame_head] = 0x1;
    frame_head = (frame_head + 1) % BUFF_SIZE;
    pthread_cond_broadcast(&monitor.child_thread_cond);


    // Handle User Exit Input
    user_exit = monitor.eof;

    // Unlock Parent Thread Mutex
    pthread_mutex_unlock(&monitor.mutex);
  }

  // -------------------------------------------------- //

  // Once EOF has been reached (or error occurred), join threads together to end process
  for (uint8_t i = 0; i < SEG_CNT; i++) {
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

  int frame_tail = 0;
  while (1) {
    // Create Pipeline Buffer Iterator
    frame_tail = frame_tail % BUFF_SIZE;

    // Give single Child Thread permission to process
    pthread_mutex_lock(&monitor.mutex);
    
    // Put Child Thread to sleep and give Parent Thread access until new frame is processed
    while (!monitor.eof && !monitor.frame_handler_state[frame_tail] && !monitor.parent_ready) {
      pthread_cond_wait(&monitor.child_thread_cond, &monitor.mutex);      
    }

    // Kill Child Thread once EOF is reached
    if (monitor.eof) {
      pthread_mutex_unlock(&monitor.mutex);
      break;
    }

    // -------------------------------------------------- //

    // Localize individual segment data to Child Thread only if frame is available in pipeline buffer
    if (monitor.frame_handler_state[frame_tail]) {
      cv::Mat seg_in_local = monitor.frame_InUse[frame_tail].seg_in[idx];

      // Unlock Child Thread; Allow other threads to process while this thread applies filters locally
      pthread_mutex_unlock(&monitor.mutex);


      // Process Segment Mat (Apply Filters)
      cv::Mat grayscale_local = to442_grayscale(seg_in_local);
      cv::Mat sobel_local = to442_sobel(grayscale_local);


      // Update Monitor Status to Reflect Done Child Thread
      pthread_mutex_lock(&monitor.mutex);
      monitor.frame_InUse[frame_tail].seg_out[idx] = sobel_local;
      monitor.seg_done[frame_tail][idx] = 0x1;

      // Wake Output Thread if all segments for this frame are done
      if (monitor.seg_done[frame_tail][0] && monitor.seg_done[frame_tail][1] && monitor.seg_done[frame_tail][2] && monitor.seg_done[frame_tail][3]) {
        pthread_cond_signal(&monitor.output_thread_cond);
      }

      // Increment pipeline buffer index
      frame_tail++;
    }

    // Unlock Child Thread; Allow other threads to process
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
    // Create Pipeline Buffer Iterator
    static int frame_tail = 0;
    frame_tail = frame_tail % BUFF_SIZE;

    // Give access to Output Thread
    pthread_mutex_lock(&monitor.mutex);

    while ((!monitor.eof) && !(monitor.frame_handler_state[frame_tail] &&
          monitor.seg_done[frame_tail][0] && monitor.seg_done[frame_tail][1] &&
          monitor.seg_done[frame_tail][2] && monitor.seg_done[frame_tail][3])) {
      // Wait for Parent Thread to receive confirmation of ALL Child Thread completion
      pthread_cond_wait(&monitor.output_thread_cond, &monitor.mutex);
    }

    // Kill Output Thread if EOF
    if (monitor.eof) {
      pthread_mutex_unlock(&monitor.mutex);
      break;
    }


    // -------------------------------------------------- //
   
    // Run Output Thread process only if frame is available in pipeline buffer
    cv::Mat seg_out_local[SEG_CNT];
    static uint8_t idx_lock = 0x0;
    static int full_width, full_height;
    static cv::Rect remove_padding[SEG_CNT];
    static cv::Rect ref_segs_local[SEG_CNT];

    if (monitor.frame_handler_state[frame_tail]) {
      // Save local copy of segment data
      for (uint8_t i = 0; i < SEG_CNT; i++) {
        seg_out_local[i] = monitor.frame_InUse[frame_tail].seg_out[i];
      }

      // Recover Full Frame Size from Reference Segments (Only run once)
      if (!idx_lock) {
        idx_lock++;

        full_width = monitor.ref_segs[0].width;
        full_height = (monitor.ref_segs[3].y + monitor.ref_segs[3].height);

        // Create Parameters to Stitch Sobel Segments together (Remove Padding) 
        for (uint8_t i = 0; i < SEG_CNT; i++) {
          // Save local copy of Reference Segments
          ref_segs_local[i] = monitor.ref_segs[i];

          // topPadding = 0 for Top Segment (i = 0), topPadding = 1 otherwise
          // bottomPadding = 0 for Bottom Segment (i = 3), bottomPadding = 1 otherwise
          int topPadding = monitor.ref_segs[i].y - monitor.worker_segs[i].y;
          int bottomPadding = (monitor.worker_segs[i].y + monitor.worker_segs[i].height) - (monitor.ref_segs[i].y + monitor.ref_segs[i].height); 

          remove_padding[i] = cv::Rect(0, topPadding, seg_out_local[i].cols, seg_out_local[i].rows - topPadding - bottomPadding);
        }
      }
  
      // Clear frame from pipeline buffer and increment buffer Iterator
      monitor.frame_handler_state[frame_tail] = 0x0;
      for (uint8_t i = 0; i < SEG_CNT; i++) {
        monitor.seg_done[frame_tail][i] = 0x0;
      }
      pthread_cond_signal(&monitor.parent_thread_cond);
      frame_tail++; 

      // Allow other threads to pass while output is being prepared
      pthread_mutex_unlock(&monitor.mutex);

      // Allocate Memory for Output Frame
      cv::Mat outFrame(full_height, full_width, CV_8UC1);

      // Write processed segments to Output Frame
      for (uint8_t i = 0; i < SEG_CNT; i++) {
        seg_out_local[i](remove_padding[i]).copyTo(outFrame(ref_segs_local[i]));
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
        pthread_cond_signal(&monitor.parent_thread_cond);
        pthread_mutex_unlock(&monitor.mutex);

        // End Output Thread
        break;
      }


      // Relock all other threads before finishing
      pthread_mutex_lock(&monitor.mutex);
    }

    // Allow other threads to pass
    pthread_mutex_unlock(&monitor.mutex); 
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


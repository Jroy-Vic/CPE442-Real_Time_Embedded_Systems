/*
 * Roy Vicerra (rvicerra)
 * CPE 442-01
 * Lab 6
*/


#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <papi.h>
#include <chrono>
#include <sched.h>
#include "filters.hpp"

// MACROS:
#define SEG_CNT     4
#define BUFF_SIZE   100

// Thread Prototypes:
void* inputParse_Thread1(void* vid_ptr);
void* processSegment_Thread2(void* seg_idx_ptr);
void* recombFrame_Thread3(void*);

// Core Functions:
// Isolate Function to a single core
void pin_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
  }
}


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

// Create Global Frame Counters
uint32_t frameCnt_thread1 = 0;
uint32_t frameCnt_thread2 = 0;
uint32_t frameCnt_thread3 = 0;
long long values_thread1[2] = {0,0};
long long values_thread2[2] = {0,0};
long long values_thread3[2] = {0,0};

// ------------------------------------------------------- //
// Thread Definitions //

// Thread Function: Pull in frames from a video file/webcam one after another,
// splitting each frame into four equal rows before moving on to the next;
// apply padding pixels when necessary and pass each segment to child threads.
void* inputParse_Thread1(void* vid_ptr) {
  // Isolate Thread1 to Core 0 
  pin_to_core(0);
  // Compute PAPI Characteristics
  PAPI_register_thread();

  int eventSet = PAPI_NULL;
  int counters[2] = {PAPI_TOT_CYC, PAPI_L1_DCM};

  PAPI_create_eventset(&eventSet);
  PAPI_add_events(eventSet, counters, 2);
  PAPI_start(eventSet);


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

    // Increment Frame Counter
    frameCnt_thread1++;

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
    pthread_cond_broadcast(&monitor.child_thread_cond);

    // Handle User Exit Input
    user_exit = monitor.eof;

    // Unlock Parent Thread Mutex
    pthread_mutex_unlock(&monitor.mutex);

    // Increment Frame Head
    frame_head = (frame_head + 1) % BUFF_SIZE;
  }

  // -------------------------------------------------- //
  
  // Calculate PAPI Characteristics
  PAPI_stop(eventSet, values_thread1);
  PAPI_cleanup_eventset(eventSet);
  PAPI_destroy_eventset(&eventSet);
  PAPI_unregister_thread();

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

  // Isolate Thread2 (Child 0 Only) to Core 1
  int eventSet, counters[2];
  if (idx == 0) {
    pin_to_core(1);

    PAPI_register_thread();

    eventSet = PAPI_NULL;
    counters[0] = PAPI_TOT_CYC;
    counters[1] = PAPI_L1_DCM;
    PAPI_create_eventset(&eventSet);
    PAPI_add_events(eventSet, counters, 2);
    PAPI_start(eventSet);
  }


  int frame_tail = 0;
  while (1) {
    // Create Pipeline Buffer Iterator and iterate PAPI Counter
    frame_tail = frame_tail % BUFF_SIZE;
    if (idx == 0) {
      frameCnt_thread2++;
    }

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


  // Calculate PAPI Characteristics for Child 0
  if (idx == 0) {
    PAPI_stop(eventSet, values_thread2);
    PAPI_cleanup_eventset(eventSet);
    PAPI_destroy_eventset(&eventSet);
    PAPI_unregister_thread();
  }
  
  std::cout << "Child Thread " << idx << " done.\n";
  return nullptr;
}


// Thread Function: Wait for all four frame segments to be processed and received,
// then stitch them all back into a single frame.
void* recombFrame_Thread3(void*) {
  // Isolate Thread3 to Core 2 
  pin_to_core(2);
  // Start PAPI Counter
  PAPI_register_thread();

  int eventSet = PAPI_NULL;
  int counters[2] = {PAPI_TOT_CYC, PAPI_L1_DCM};
  PAPI_create_eventset(&eventSet);
  PAPI_add_events(eventSet, counters, 2);
  PAPI_start(eventSet);


  // Create a Single Display Window
  cv::namedWindow("Sobel", cv::WINDOW_AUTOSIZE);

  while (1) {
    // Create Pipeline Buffer Iterator and iterate PAPI Counter
    static int frame_tail = 0;
    frame_tail = frame_tail % BUFF_SIZE;
    frameCnt_thread3++;

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

  
  // Stop PAPI Counter
  PAPI_stop(eventSet, values_thread3);
  PAPI_cleanup_eventset(eventSet);
  PAPI_destroy_eventset(&eventSet);
  PAPI_unregister_thread();

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

  // Initialize PAPI
  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
    std::cerr << "PAPI_library_init failed\n";

    return -1;
  }
  if (PAPI_thread_init((unsigned long (*)(void)) pthread_self) != PAPI_OK) {
    std::cerr << "PAPI_thread_init failed\n";

    return -1;
  }

  int eventSet = PAPI_NULL;
  int counters[2] = {PAPI_TOT_CYC, PAPI_L1_DCM};
  long long values[2] = {0,0};

  if (PAPI_create_eventset(&eventSet) != PAPI_OK) {
    std::cerr << "PAPI_create_eventset failed\n";

    return -1;
  }
  if (PAPI_add_events(eventSet, counters, 2) != PAPI_OK) {
    std::cerr << "PAPI_add_events failed\n";

    return -1;
  } 

  // Start PAPI Counters
  if (PAPI_start(eventSet) != PAPI_OK) {
    std::cerr << "PAPI_start failed\n";

    return -1;
  }
  auto timeStart = std::chrono::steady_clock::now();


  // Create Main Parent Thread and Output Thread
  pthread_t parent_thread, output_thread;
  void* parent_thread_status, *output_thread_status;
  pthread_create(&parent_thread, NULL, inputParse_Thread1, &vid);
  pthread_create(&output_thread, NULL, recombFrame_Thread3, NULL);

  // Wait for Parent Thread and Output Thread to Finish
  pthread_join(parent_thread, &parent_thread_status);
  pthread_join(output_thread, &output_thread_status);


  // Stop PAPI Counters
  auto timeEnd = std::chrono::steady_clock::now();
  if (PAPI_stop(eventSet, values) != PAPI_OK) {
    std::cerr << "PAPI_stop failed\n";

    return -1;
  }


  // Close OpenCV Applications
  vid.release();
  cv::destroyAllWindows();


  // Print Counter Data
  double timeTotal = std::chrono::duration<double>(timeEnd - timeStart).count();
  std::cout << "\n// ---------- PAPI Counter Data ---------- //\n\n"; 
  std::cout << "Total Time Elapsed (sec): " << timeTotal << "\n";
  std::cout << "Total Cycles Processed: " << values[0] << "\n";
  std::cout << "Total L1 Cache Misses: " << values[1] << "\n";
  std::cout << "Total Average Frames per Second (FPS): " << (frameCnt_thread1 / timeTotal) << "\n\n";
  std::cout << "// ---------- Thread 1, Core 0 ----------- //\n\n";
  std::cout << "Average Number of Cache Misses per Frame: " << (values_thread1[1] / frameCnt_thread1) << "\n";
  std::cout << "Average Number of Cycles per Frame: " << (values_thread1[0] / frameCnt_thread1) << "\n\n";
  std::cout << "// --- Thread 2, Core 1 (Child 0 Only) --- //\n\n";
  std::cout << "Average Number of Cache Misses per Frame: " << (values_thread2[1] / frameCnt_thread2) << "\n";
  std::cout << "Average Number of Cycles per Frame: " << (values_thread2[0] / frameCnt_thread2) << "\n\n";
  std::cout << "// ---------- Thread 3, Core 2 ----------- //\n\n";
  std::cout << "Average Number of Cache Misses per Frame: " << (values_thread3[1] / frameCnt_thread3) << "\n";
  std::cout << "Average Number of Cycles per Frame: " << (values_thread3[0] / frameCnt_thread3) << "\n\n";
  std::cout << "// --------------------------------------- //\n\n";
  PAPI_cleanup_eventset(eventSet);
  PAPI_destroy_eventset(&eventSet);
  PAPI_shutdown();

  return 0;
}


# Roy Vicerra (rvicerra)
# CPE 442-01
# Lab 3


CC							= g++
CFLAGS 					= -g -Wall -Wextra -Werror -std=c++11 -O0 -lpthread
OPENCV_CFLAGS 	= $(shell pkg-config --cflags opencv4)
OPENCV_LIBS 		= $(shell pkg-config --libs opencv4)
PTHREAD_LIBS		= -lpthread
LD 							= $(CC)
LDFLAGS 				= $(CFLAGS)

TARGET	= sobelFilter
SRC			= main.cpp filters.cpp
HDR 		= filters.hpp
OBJ			= $(SRC:.cpp=.o)


default: all

all: $(TARGET)

$(TARGET): $(OBJ)
	$(LD) $(LDFLAGS) -o $@ $^ $(OPENCV_LIBS) $(PTHREAD_LIBS)

%.o: %.cpp $(HDR)
		$(CC) $(CFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ)



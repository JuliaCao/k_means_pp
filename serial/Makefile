CC = g++ -std=c++11
//LDLIBS = -openmp
target = serial

all: serial serial_knl

serial: serial.cpp
	$(CC) serial.cpp -o serial $(LDLIBS)

serial_knl: serial.cpp
	module swap craype-haswell craype-mic-knl
	$(CC) serial.cpp -o serial_knl $(LDLIBS)
	module swap craype-mic-knl craype-haswell

clean:
	rm -f $(target)


CXX = CC

# upcxx-meta PPFLAGS are really CFLAGS to be used during compilation
# upcxx-meta LDFLAGS are really CFLAGS to be used during linking
# upcxx-meta LIBFLAGS are really a combination of LDLIBS and LDFLAGS

CXXFLAGS = `upcxx-meta PPFLAGS` `upcxx-meta LDFLAGS`
LDFLAGS = `upcxx-meta LIBFLAGS`

all: upc

upc: upc.cpp
	$(CXX) upc.cpp -o upc $(CXXFLAGS) $(LDFLAGS)

clean:
	@rm -fv upc


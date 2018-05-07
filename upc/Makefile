
CXX = CC

# upcxx-meta PPFLAGS are really CFLAGS to be used during compilation
# upcxx-meta LDFLAGS are really CFLAGS to be used during linking
# upcxx-meta LIBFLAGS are really a combination of LDLIBS and LDFLAGS

CXXFLAGS = `upcxx-meta PPFLAGS` `upcxx-meta LDFLAGS`
LDFLAGS = `upcxx-meta LIBFLAGS`

all: upc upc_old

upc: upc.cpp
	$(CXX) upc.cpp -o upc $(CXXFLAGS) $(LDFLAGS)

upc_old: upc_old.cpp
	$(CXX) upc_old.cpp -o upc_old $(CXXFLAGS) $(LDFLAGS)


clean:
	@rm -fv all


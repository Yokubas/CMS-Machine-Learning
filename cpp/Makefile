CC=g++

CFLAGS=-c -g -Wall `root-config --cflags`

LDFLAGS=`root-config --glibs`

all: read_root getEntry read_root_mediumid analysis

read_root: read_root.cpp
	$(CC) $(CFLAGS) -o read_root.o read_root.cpp
	$(CC) -o read_root read_root.o $(LDFLAGS)

getEntry: getEntry.cpp
	$(CC) $(CFLAGS) -o getEntry.o getEntry.cpp
	$(CC) -o getEntry getEntry.o $(LDFLAGS)

read_root_mediumid: read_root_mediumid.cpp
	$(CC) $(CFLAGS) -o read_root_mediumid.o read_root_mediumid.cpp
	$(CC) -o read_root_mediumid read_root_mediumid.o $(LDFLAGS)

read_root_for_plot: read_root_for_plot.cpp
	$(CC) $(CFLAGS) -o read_root_for_plot.o read_root_for_plot.cpp
	$(CC) -o read_root_for_plot read_root_for_plot.o $(LDFLAGS)

analysis: analysis.cpp
	$(CC) $(CFLAGS) -o analysis.o analysis.cpp
	$(CC) -o analysis analysis.o $(LDFLAGS)

clean:
	rm -f ./*~ ./*.o ./read_root
	rm -f ./*~ ./*.o ./getEntry
	rm -f ./*~ ./*.o ./read_root_mediumid
	rm -f ./*~ ./*.o ./read_root_for_plot
	rm -f ./*~ ./*.o ./analysis
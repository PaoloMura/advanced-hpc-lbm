# Makefile

EXE=d2q9-bgk

CC=mpiicc
# CC=icc
# CC=gcc
# CFLAGS= -std=c99 -Wall -O3 -xAVX2
CFLAGS= -std=c99 -Wall -O3 -xAVX2 -g -qopt-report=5 -simd -vec -qopenmp
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)

# Export roofline to HTML
roofline:
	# advixe-cl --report=roofline --project-dir=./advi_results1 --report-output=./roofline1.html
	# advixe-cl --report=roofline --project-dir=./advi_results2 --report-output=./roofline2.html
	# advixe-cl --report=roofline --project-dir=./advi_results3 --report-output=./roofline3.html
	advixe-cl --report=roofline --project-dir=./advi_results4 --report-output=./roofline4.html
	advixe-cl --report=roofline --project-dir=./advi_results5 --report-output=./roofline5.html
	advixe-cl --report=roofline --project-dir=./advi_results6 --report-output=./roofline6.html

tripcounts:
	advixe-cl --collect=survey --project-dir=./advixe_project7 \
	-- mpiicc -std=c99 -Wall -O3 -xAVX2 -g -qopt-report=5 -simd -vec -qopenmp d2q9-bgk.c


mpi_roofline:
	advixe-cl --report=roofline --project-dir=advi_results_4/project1 --report-output=./mpi_roofline.html

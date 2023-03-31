# Makefile

# EXE=d2q9-bgk
EXE=test

# CC=icc
# CC=gcc
CC=mpiicc
CFLAGS= -std=c99 -Wall -O3
# CFLAGS= -std=c99 -Wall -O3 -g -qopt-report=5
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
	advixe-cl --report=roofline --project-dir=./advi_results1 --report-output=./roofline1.html
	# advixe-cl --report=roofline --project-dir=./advi_results2 --report-output=./roofline2.html
	# advixe-cl --report=roofline --project-dir=./advi_results3 --report-output=./roofline3.html

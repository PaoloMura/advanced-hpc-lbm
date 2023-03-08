# Makefile

EXE=d2q9-bgk

CC=icc
# CC=gcc
CFLAGS= -std=c99 -Wall -O3 -xAVX2 -qopenmp
# CFLAGS= -std=c99 -Wall -g -O3 -xAVX2 -qopenmp -qopt-report=5 -qopt-report-phase=vec
# CFLAGS= -std=c99 -Wall -O3 -xAVX2 -qopenmp -g -qopt-report=5 -simd
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/1024x1024.final_state.dat
REF_AV_VELS_FILE=check/1024x1024.av_vels.dat

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
	advixe-cl --report=roofline --project-dir=./advi_results --report-output=./roofline.html

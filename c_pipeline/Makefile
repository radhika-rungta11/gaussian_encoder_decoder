CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -Wno-unused-parameter -std=c99
LDFLAGS ?=
LIBS    ?= -lz -lm

SRCS = main.c \
       decoder.c \
       encoder.c \
       renderer_cpu.c \
       scene.c \
       huffman.c \
       rvq.c \
       fp16.c \
       gzip_io.c

OBJS = $(SRCS:.c=.o)

BIN = 4dgs_pipeline

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(BIN)

.PHONY: all clean

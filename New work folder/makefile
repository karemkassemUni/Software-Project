CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS = -lm

all: symnmf

symnmf: symnmf.o
	$(CC) $(CFLAGS) symnmf.o -o symnmf $(LDFLAGS)

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

clean:
	rm -f *.o symnmf

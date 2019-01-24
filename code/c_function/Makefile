CFLAGS := -fPIC -Wall -Werror -std=c99 -fopenmp
CC := gcc
NAME := flatsreg_core

lib: $(NAME).so

$(NAME).so: $(NAME).o
	$(CC) -shared -fPIC -Wall -std=c99 -Wl,-soname,$(NAME).so $^ -fopenmp -o $@
	
	-rm -r *.o*
	
clean:
	$(RM) *.o *.so*

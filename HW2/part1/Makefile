all: pi

pi: pi.cpp 
	g++ -fPIC -O3 -mavx2 -mavx -mfma -Wall -pedantic pi.cpp -o pi.out -lpthread

clean:
	rm -f *.o *.s myexp *~

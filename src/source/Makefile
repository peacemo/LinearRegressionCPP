linearRegression: main.o linearRegression.o utils.o
	g++ main.o linearRegression.o utils.o -o linearRegression

main.o: main.cpp
	g++ -c main.cpp

linearRegression.o: linearRegression.cpp
	g++ -c linearRegression.cpp

utils.o: utils.cpp
	g++ -c utils.cpp

clean:
	rm -f *.o linearRegression

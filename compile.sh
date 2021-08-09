gcc -std=c17 -I/home/johannes/src/volk/include -x c main.c -o mainvolkgnuc -lm
clang -std=c17 -I/home/johannes/src/volk/include -x c main.c -o mainvolkclangc -lm
g++ -std=c++17 -I/home/johannes/src/volk/include -x c++ main.cc -o mainvolkcpp -lm -lfmt
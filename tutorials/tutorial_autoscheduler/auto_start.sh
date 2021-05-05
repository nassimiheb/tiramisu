export TIRAMISU_ROOT=/data/scratch/tiramisu_git/tiramisu_lanka/tiramisu
export LD_LIBRARY_PATH=/data/scratch/tiramisu_git/tiramisu_lanka/tiramisu/3rdParty/Halide/lib

rm -dr ${TIRAMISU_ROOT}/tutorials/tutorial_autoscheduler/retry$1/build/*

cd ${TIRAMISU_ROOT}/tutorials/tutorial_autoscheduler/retry$1/build
echo trying program auto_scheduler $1
sed -i -E "s/perform_autoscheduling=(false|true)/perform_autoscheduling=false/gi" ${TIRAMISU_ROOT}/tutorials/tutorial_autoscheduler/retry$1/generator.cpp
cmake ..
make
../generator > ${TIRAMISU_ROOT}/tutorials/tutorial_autoscheduler/log$1.txt
g++ -shared -o function.o.so function.o
g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/benchmarks -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -o wrapper  -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ../wrapper.cpp ./function.o.so -ltiramisu -lHalide -ldl -lpthread -lz -lm

sed -i -E "s/perform_autoscheduling=(false|true)/perform_autoscheduling=true/gi" ${TIRAMISU_ROOT}/tutorials/tutorial_autoscheduler/retry$1/generator.cpp
cmake ..
make
echo "======================step2==================================" >> ${TIRAMISU_ROOT}/tutorials/tutorial_autoscheduler/log$1.txt
../generator >> ${TIRAMISU_ROOT}/tutorials/tutorial_autoscheduler/log$1.txt



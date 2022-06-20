num_thread=${1:-1}

mkdir build

cd build

mkdir ilqr_planner
mkdir pylqr_planner

cd ilqr_planner
cmake ../../ilqr_planner -DCMAKE_BUILD_TYPE=Release
make -j $num_thread
sudo make install

cd ../pylqr_planner
cmake ../../pylqr_planner -DCMAKE_BUILD_TYPE=Release
make -j $num_thread
sudo make install

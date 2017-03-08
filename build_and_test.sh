cd gccbuild
make
cd ../pgibuild
make

./hackatron
cd ../gccbuild
./hackatron
cd ..
python signaldiff.py

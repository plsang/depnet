luarocks install nn
luarocks install nngraph
luarocks install image
luarocks install cutorch
luarocks install cunn
luarocks install loadcaffe
luarocks install lualogging

git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5 && luarocks make hdf5-0-0.rockspec && cd - && rm -rf torch-hdf5

wget http://www.kyne.com.au/~mark/software/download/lua-cjson-2.1.0.tar.gz
tar -xvf lua-cjson-2.1.0.tar.gz
cd lua-cjson-2.1.0 && luarocks make && cd - && rm -rf lua-cjson-2.1.0

git clone https://github.com/slembcke/debugger.lua.git torch-debugger
cd torch-debugger && luarocks make && cd - && rm -rf torch-debugger

git clone https://github.com/plsang/nn.git torch-nn
cd torch-nn && luarocks make rocks/nn-scm-1.rockspec && cd - && rm -rf torch-nn

git clone https://github.com/plsang/cunn.git torch-cunn
cd torch-cunn && luarocks make rocks/cunn-scm-1.rockspec && cd - && rm -rf torch-cunn


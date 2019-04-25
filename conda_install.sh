# conda install for i23seg package
export i23seg_VERSION=0.1.0
conda build recipe/ --numpy 1.14 --python 2.7  
conda install -y -q  i23-seg=${i23seg_VERSION} --use-local --force-reinstall


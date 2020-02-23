# conda install for colleda package
export VERSION=`date +%Y.%m`
conda build recipe/ --numpy 1.14 --python 2.7  
conda install -y -q  i23-seg --use-local --force-reinstall


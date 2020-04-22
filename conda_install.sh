# conda install for larix package
export VERSION=`date +%Y.%m`
conda build recipe/ --numpy 1.15 --python 3.5
conda install -c file://${CONDA_PREFIX}/conda-bld/ larix

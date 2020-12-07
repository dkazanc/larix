# Only need to change these two variables
PKG_NAME=larix
USER=dkazanc
OS=linux-64
dk-8105746b-0366-49d7-a4e9-017e5c1c41c1

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export VERSION=`date +%Y.%m`
conda install --yes anaconda-client
conda build .
#$CONDA/bin/anaconda upload -u $USER $CONDA_BLD_PATH/$OS/$PKG_NAME-`date +%Y.%m`*.tar.bz2 --force

# upload packages to conda
find $CONDA_BLD_PATH/$OS -name *.tar.bz2 | while read file
do
    echo $file
    $CONDA/bin/anaconda -v --show-traceback upload -u $USER $file --force
done

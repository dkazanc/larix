# Only need to change these two variables
PKG_NAME=larix
USER=dkazanc

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export VERSION=`date +%Y.%m`
conda build .
$CONDA/bin/conda -t $CONDA_UPLOAD_TOKEN upload -u $USER $CONDA_BLD_PATH/$PKG_NAME-`date +%Y.%m`*.tar.bz2 --force

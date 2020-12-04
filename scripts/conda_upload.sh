# Only need to change these two variables
console.log(`This platform is ${process.platform}`);
PKG_NAME=larix
USER=dkazanc

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export VERSION=`date +%Y.%m`
conda build .
$CONDA -t $CONDA_UPLOAD_TOKEN upload -u $USER $CONDA_BLD_PATH/$PKG_NAME-`date +%Y.%m`*.tar.bz2 --force

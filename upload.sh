# Only need to change these two variables
PKG_NAME=larix
USER=dkazanc

OS=linux-64
conda config --set anaconda_upload no
export CONDA_BLD_PATH=/var/lib/jenkins/.conda/envs/${BUILD_TAG}/conda-bld
export VERSION=`date +%Y.%m`
cd recipe
conda build .
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER $CONDA_BLD_PATH/$OS/$PKG_NAME-`date +%Y.%m`*.tar.bz2 --force

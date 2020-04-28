pipeline {
    agent any

    triggers {
        pollSCM('H 21 * * *')
    }

    options {
        skipDefaultCheckout(true)
        // Keep the 10 most recent builds
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }

    environment {
      PATH="/var/lib/jenkins/miniconda3/bin:$PATH"
    }

    stages {

        stage ("Code pull"){
            steps{
                checkout scm
            }
        }

        stage('Env py35') {
            steps {
                echo "Building virtualenv for py35"
                sh  ''' conda create --yes -n "${BUILD_TAG}py35" python=3.5
                        source activate "${BUILD_TAG}py35"
                        conda install cython
                        conda install conda-build
                        conda install anaconda-client
                    '''
            }
        }

        stage('Env py36') {
            steps {
                echo "Building virtualenv for py36"
                sh  ''' conda create --yes -n "${BUILD_TAG}py36" python=3.6
                        source activate "${BUILD_TAG}py36"
                        conda install cython
                        conda install conda-build
                        conda install anaconda-client
                    '''
            }
        }

        stage('Env py37') {
            steps {
                echo "Building virtualenv for py37"
                sh  ''' conda create --yes -n "${BUILD_TAG}py37" python=3.7
                        source activate "${BUILD_TAG}py37"
                        conda install cython
                        conda install conda-build
                        conda install anaconda-client
                    '''
            }
        }

        stage('Build larix-py35') {
            when {
                expression {
                    currentBuild.result == null || currentBuild.result == 'SUCCESS'
                }
            }
            steps {
                sh  ''' source activate "${BUILD_TAG}py35"
                        conda config --set anaconda_upload no
                        export VERSION=`date +%Y.%m`
                        conda build recipe/ --numpy 1.15 --python 3.5
                        conda install --yes -c file://${CONDA_PREFIX}/conda-bld/ larix
                    '''
            }
        }

        stage('Build larix-py36') {
            when {
                expression {
                    currentBuild.result == null || currentBuild.result == 'SUCCESS'
                }
            }
            steps {
                sh  ''' source activate "${BUILD_TAG}py36"
                        conda config --set anaconda_upload no
                        export VERSION=`date +%Y.%m`
                        conda build recipe/ --numpy 1.15 --python 3.6
                        conda install --yes -c file://${CONDA_PREFIX}/conda-bld/ larix
                    '''
            }
        }

        stage('Build larix-py37') {
            when {
                expression {
                    currentBuild.result == null || currentBuild.result == 'SUCCESS'
                }
            }
            steps {
                sh  ''' source activate "${BUILD_TAG}py37"
                        conda config --set anaconda_upload no
                        export VERSION=`date +%Y.%m`
                        conda build recipe/ --numpy 1.15 --python 3.7
                        conda install --yes -c file://${CONDA_PREFIX}/conda-bld/ larix
                    '''
            }
        }

        stage('Run tests py35') {
            steps {
                sh  ''' source activate "${BUILD_TAG}py35"
                        conda install pytest pytest-cov
                        pytest -v --cov tests/
                    '''
            }
        }

        stage('Run tests py36') {
            steps {
                sh  ''' source activate "${BUILD_TAG}py36"
                        conda install pytest pytest-cov
                        pytest -v --cov tests/
                    '''
            }
        }

        stage('Run tests py37') {
            steps {
                sh  ''' source activate "${BUILD_TAG}py37"
                        conda install pytest pytest-cov
                        pytest -v --cov tests/
                    '''
            }
        }

        stage("Deploy py35") {
             steps {
                 sh ''' source activate "${BUILD_TAG}py35"
                        conda config --set anaconda_upload yes
                        source /var/lib/jenkins/upload.sh
                        anaconda -t $CONDA_UPLOAD_TOKEN upload -u dkazanc /var/lib/jenkins/.conda/envs/"${BUILD_TAG}py35"/conda-bld/linux-64/*.tar.bz2 --force
                    '''
             }
        }
        stage("Deploy py36") {
             steps {
                 sh ''' source activate "${BUILD_TAG}py36"
                        conda config --set anaconda_upload yes
                        source /var/lib/jenkins/upload.sh
                        anaconda -t $CONDA_UPLOAD_TOKEN upload -u dkazanc /var/lib/jenkins/.conda/envs/"${BUILD_TAG}py36"/conda-bld/linux-64/*.tar.bz2 --force
                    '''
             }
        }
        stage("Deploy py37") {
             steps {
                 sh ''' source activate "${BUILD_TAG}py37"
                        conda config --set anaconda_upload yes
                        source /var/lib/jenkins/upload.sh
                        anaconda -t $CONDA_UPLOAD_TOKEN upload -u dkazanc /var/lib/jenkins/.conda/envs/"${BUILD_TAG}py37"/conda-bld/linux-64/*.tar.bz2 --force
                    '''
             }
        }
    }

    post {
        always {
            sh ''' conda remove --yes -n "${BUILD_TAG}py35" --all
                   conda remove --yes -n "${BUILD_TAG}py36" --all
                   conda remove --yes -n "${BUILD_TAG}py37" --all
               '''
        }
        failure {
            emailext (
                subject: "FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                body: """<p>FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]':</p>
                         <p>Check console output at &QUOT;<a href='${env.BUILD_URL}'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a>&QUOT;</p>""",
                recipientProviders: [[$class: 'DevelopersRecipientProvider']])
        }
    }

}

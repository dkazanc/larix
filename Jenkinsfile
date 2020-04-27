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

        stage('Build environment') {
            steps {
                echo "Building virtualenv"
                sh  ''' conda create --yes -n ${BUILD_TAG} python=3.7
                        source activate ${BUILD_TAG}
                        conda install cython
                        conda install conda-build
                        conda install anaconda-client
                    '''
            }
        }

        stage('Build package') {
            when {
                expression {
                    currentBuild.result == null || currentBuild.result == 'SUCCESS'
                }
            }
            steps {
                sh  ''' source activate ${BUILD_TAG}
                        export VERSION=`date +%Y.%m`
                        conda build recipe/ --numpy 1.15 --python 3.7
                        conda install --yes -c file://${CONDA_PREFIX}/conda-bld/ larix
                    '''
            }
        }

        stage('Unit tests') {
            steps {
                sh  ''' source activate ${BUILD_TAG}
                        python -m unittest tests/test.py --verbose
                    '''
            }
        }

        stage("Upload to anaconda") {
             steps {
                 sh ''' source activate ${BUILD_TAG}
                        anaconda -t $CONDA_UPLOAD_TOKEN upload -u dkazanc /var/lib/jenkins/.conda/envs/${BUILD_TAG}/conda-bld/linux-64/*tar.bz2 --force
                    '''
             }
        }
    }

    post {
        always {
            sh 'conda remove --yes -n ${BUILD_TAG} --all'
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

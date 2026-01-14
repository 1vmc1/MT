pipeline {
    agent { label 'vmc2' }

    environment {
        OS_CLOUD = 'mycloud'
        // TELEGRAM_TOKEN — должен быть сохранён в Jenkins (Secret text)
        TELEGRAM_TOKEN = credentials('TELEGRAM_TOKEN')
        // Не используем GitHub token — репозиторий публичный
    }

    stages {
        stage('Terraform Apply') {
            steps {
                dir('terraform') {
                    sh '''
                        set -e
                        terraform init
                        terraform apply -auto-approve \
                          -var="image_name=ubuntu-20.04" \
                          -var="flavor_name=m1.medium" \
                          -var="network_name=sutdents-net" \
                          -var="keypair=jenkins-key"
                        terraform output -raw vm_ip > ../ansible/ip.txt
                    '''
                }
            }
        }

        stage('Wait for SSH') {
            steps {
                dir('ansible') {
                    sh '''
                        set -e
                        VM_IP=$(cat ip.txt)
                        echo "Waiting for SSH on ${VM_IP}..."
                        for i in {1..30}; do
                          if nc -z ${VM_IP} 22; then
                            echo "SSH is ready"
                            exit 0
                          fi
                          sleep 5
                        done
                        echo "SSH did not become available"
                        exit 1
                    '''
                }
            }
        }

        stage('Install Docker on VM') {
            steps {
                dir('ansible') {
                    sh '''
                        set -e
                        VM_IP=$(cat ip.txt)
                        ssh -o StrictHostKeyChecking=no -i ~/.ssh/jenkins_deploy_rsa ubuntu@${VM_IP} bash -s <<'EOF'
set -e
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release python3-pip software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker ubuntu || true
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose || true
sudo chmod +x /usr/local/bin/docker-compose || true
sudo pip3 install docker || true
EOF
                    '''
                }
            }
        }

        stage('Docker Login') {
            steps {
                dir('ansible') {
                    // Замените 'docker-hub-cred' на ваш credentialsId (username/password)
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-cred', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                        sh '''
                            set -e
                            VM_IP=$(cat ip.txt)
                            ssh -o StrictHostKeyChecking=no -i ~/.ssh/jenkins_deploy_rsa ubuntu@${VM_IP} "echo \$DOCKER_PASS | sudo docker login --username \$DOCKER_USER --password-stdin"
                        '''
                    }
                }
            }
        }

        stage('Deploy with Ansible') {
            steps {
                dir('ansible') {
                    sh '''
                        set -e
                        VM_IP=$(cat ip.txt)
                        echo "[servers]" > inventory.ini
                        echo "vm ansible_host=${VM_IP} ansible_user=ubuntu ansible_ssh_private_key_file=~/.ssh/jenkins_deploy_rsa" >> inventory.ini

                        ansible-playbook -i inventory.ini \
                          --ssh-common-args='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
                          -e "telegram_token=${TELEGRAM_TOKEN}" \
                          playbook.yml
                    '''
                }
            }
        }
    }

    post {
        failure {
            echo 'Build failed — проверьте логи.'
        }
    }
}

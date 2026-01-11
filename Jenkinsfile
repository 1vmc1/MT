pipeline {
    agent { label 'vmc2' }

    environment {
        OS_CLOUD = 'mycloud'
        TELEGRAM_TOKEN = credentials('TELEGRAM_TOKEN')
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
                          -var="flavor_name=m1.large" \
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
                        exit 1
                    '''
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
                        ansible-playbook \
                          -i inventory.ini \
                          --ssh-common-args='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
                          -e "telegram_token=${TELEGRAM_TOKEN}" \
                          playbook.yml
                    '''
                }
            }
        }

        stage('Verify') {
            steps {
                echo '✓ Deployment successful!'
            }
        }
    }
}

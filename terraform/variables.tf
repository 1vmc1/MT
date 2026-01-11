variable "image_name" {
  type    = string
  default = "ubuntu-22.04"
}

variable "flavor_name" {
  type    = string
  default = "m1.small"
}

variable "network_name" {
  type    = string
  default = "students-net"
}

variable "keypair" {
  type    = string
  default = "jenkins-key"
}

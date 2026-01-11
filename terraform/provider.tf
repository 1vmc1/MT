terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.54.1"
    }
  }

  required_version = ">= 1.3.0"
}

provider "openstack" {}

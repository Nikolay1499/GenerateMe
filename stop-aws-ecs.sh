#!/bin/bash

#stop the containers
ecs-cli compose service rm --cluster-config generateme --ecs-profile generateme-profile

#take down your cluster
ecs-cli down --force --cluster-config generateme --ecs-profile generateme-profile
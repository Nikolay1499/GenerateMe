#!/bin/bash

# configure
ecs-cli configure --cluster generateme --default-launch-type EC2 --config-name generateme --region eu-west-2

# setup cloud formation template
ecs-cli up --keypair ecs --capability-iam  --force --size 1 --instance-type t3.small --cluster-config generateme --ecs-profile generateme-profile

#wait for the cluster
sleep 30

# deploy
ecs-cli compose -f docker-compose-aws.yml service up --create-log-groups --cluster-config generateme --ecs-profile generateme-profile

# check
ecs-cli ps --cluster-config generateme --ecs-profile generateme-profile
# Fred the robot arm
This is my latest robot arm and his name is Fred

To run tests use nosetests

# Reinforcement learning
build with: docker build -t thomas137/sac -f SaCDockerFile .

Run with:
docker run -it --gpus all --rm -v /home/thomas/PycharmProjects/fred/src:/tf/src   thomas137/sac

once in docker:
python src/reinforcementlearning/softActorCritic/soft_actor_critic.py --root_dir=test
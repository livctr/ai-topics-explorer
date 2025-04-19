LOCAL_VOLUME="cs-topics-explorer_db_data"
REMOTE_USER="ubuntu"
REMOTE_HOST="ec2-34-238-247-252.compute-1.amazonaws.com"
SSH_KEY_PEM_FILE="~/.ssh/cs-topics-explorer-server-key.pem"

# Ensures data directory exists
mkdir -p ~/data


docker run --rm \
  -v $LOCAL_VOLUME:/volume \
  -v $HOME/data:/backup \
  alpine \
  tar czf /backup/db_volume_backup.tar -C /volume .

scp -i $SSH_KEY_PEM_FILE $HOME/data/db_volume_backup.tar $REMOTE_USER@$REMOTE_HOST:~/data/db_volume_backup.tar


ssh -i $SSH_KEY_PEM_FILE $REMOTE_USER@$REMOTE_HOST \
  "docker run --rm \
  -v cs-topics-explorer_db_data:/volume \
  -v ~/data:/backup \
  alpine \
  sh -c 'tar xzf /backup/db_volume_backup.tar -C /volume';
  cd app/;
  docker compose up --build;
  "

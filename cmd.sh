python scripts/evaluate.py \
  --algorithm ppo \
  --model outputs/waypoint_ppo_1m/ppo_WaypointLunarLander-v0.zip \
  --env WaypointLunarLander-v0 \
  --vec-normalize outputs/waypoint_ppo_1m/vec_normalize.pkl

python scripts/plot_training.py \
  --monitor outputs/waypoint_ppo_1m/monitor/train.monitor.csv \
  --output reports/waypoint_ppo_1m.png

python3 scripts/render_env.py \
--env WaypointLunarLander-v0 \
--policy model \
--algorithm ppo \
--model outputs/waypoint_ppo_1m/ppo_WaypointLunarLander-v0.zip \
--vec-normalize outputs/waypoint_ppo_1m/vec_normalize.pkl \
--output outputs/waypoint_ppo_1m/waypoint_ppo.gif \
--steps 1000 \
--seed 7
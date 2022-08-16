# NYU Eval 5 Red Team Submission

## Info from NYU

The 8 scenes are in the corresponding folder

Install requirements.txt with `pip install -r requirements.txt`

Arguments passed with argparse and can run like 

```
python $task/main.py --scene_path=$scene_path
```

## Info from CACI

Download the models from S3 and copy them into the corresponding folders:
- [agent_id/best_agent.py](https://nyu-models.s3.amazonaws.com/eval-5/agent_id/best_agent.pt)
- [agent_id/ball.py](https://nyu-models.s3.amazonaws.com/eval-5/agent_id/ball.pt)
- [lava/best10.rt](https://nyu-models.s3.amazonaws.com/eval-5/lava/best10.pt)
- [moving_target/best_mt.rt](https://nyu-models.s3.amazonaws.com/eval-5/moving_target/best_mt.pt)
- [ramp/best (17).rt](https://nyu-models.s3.amazonaws.com/eval-5/ramp/best+(17).pt)
- [ramp/best10.rt](https://nyu-models.s3.amazonaws.com/eval-5/ramp/best10.pt)
- [solidity/best4.rt](https://nyu-models.s3.amazonaws.com/eval-5/solidity/best4.pt)
- [spatial_elimination/best.rt](https://nyu-models.s3.amazonaws.com/eval-5/spatial_elimination/best.pt)
- [support_relation/best4.rt](https://nyu-models.s3.amazonaws.com/eval-5/support_relation/best4.pt)
- [tool_scene/best_tool21.rt](https://nyu-models.s3.amazonaws.com/eval-5/tool_scene/best_tool21.pt)

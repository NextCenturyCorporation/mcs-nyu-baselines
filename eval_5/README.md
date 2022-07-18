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
- [lava/best.rt](https://nyu-models.s3.amazonaws.com/eval-5/lava/best.pt)
- [models/best.rt](https://nyu-models.s3.amazonaws.com/eval-5/models/best.pt)
- [models/best_v9.rt](https://nyu-models.s3.amazonaws.com/eval-5/models/best_v9.pt)
- [models/best_v11.rt](https://nyu-models.s3.amazonaws.com/eval-5/models/best_v11.pt)
- [solidity/best.rt](https://nyu-models.s3.amazonaws.com/eval-5/solidity/best.pt)
- [spatial_elimination/best.rt](https://nyu-models.s3.amazonaws.com/eval-5/spatial_elimination/best.pt)
- [support_relation/best4.rt](https://nyu-models.s3.amazonaws.com/eval-5/support_relation/best4.pt)

import argparse
import sys

import machine_common_sense as mcs

import baseline_collision
import baseline_gravity
import baseline_objectpermanence
import baseline_reori
import baseline_shapeconstancy
import baseline_spatialTemporalContinuity

# Eval-4-specific scene filename prefixes (will change in Eval 5+)
INTERACTIVE_REORIENTATION_PREFIX = 'bravo'
PASSIVE_COLLISION_PREFIX = 'november'
PASSIVE_GRAVITY_SUPPORT_PREFIX = 'oscar'
PASSIVE_OBJECT_PERMANENCE_PREFIX = 'papa'
PASSIVE_SHAPE_CONSTANCY_PREFIX = 'quebec'
PASSIVE_SPATIO_TEMPORAL_CONTINUITY_PREFIX = 'romeo'

UNITY_APP = '/home/ubuntu/unity_app/MCS-AI2-THOR-Unity-App-v0.4.7.x86_64'


def main(scene_file: str):
    scene_data, status = mcs.load_scene_json_file(scene_file)
    if status is not None:
        sys.exit(status)
    scene_name = scene_data['name']
    if scene_name.startswith(INTERACTIVE_REORIENTATION_PREFIX):
        print(
            f'Running interactive spatial reorientation baseline on '
            f'{scene_name}'
        )
        return baseline_reori.main(scene_data, UNITY_APP)
    if scene_name.startswith(PASSIVE_COLLISION_PREFIX):
        print(f'Running passive collision baseline on {scene_name}')
        return baseline_collision.main(scene_data, UNITY_APP)
    if scene_name.startswith(PASSIVE_GRAVITY_SUPPORT_PREFIX):
        print(f'Running passive gravity support baseline on {scene_name}')
        return baseline_gravity.main(scene_data, UNITY_APP)
    if scene_name.startswith(PASSIVE_OBJECT_PERMANENCE_PREFIX):
        print(f'Running passive object permanence baseline on {scene_name}')
        return baseline_objectpermanence.main(scene_data, UNITY_APP)
    if scene_name.startswith(PASSIVE_SHAPE_CONSTANCY_PREFIX):
        print(f'Running passive shape constancy baseline on {scene_name}')
        return baseline_shapeconstancy.main(scene_data, UNITY_APP)
    if scene_name.startswith(PASSIVE_SPATIO_TEMPORAL_CONTINUITY_PREFIX):
        print(
            f'Running passive spatio temporal continuity baseline on '
            f'{scene_name}'
        )
        return baseline_spatialTemporalContinuity.main(scene_data, UNITY_APP)
    sys.exit(f'No baseline to run {scene_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene_file', help='JSON scene file path')
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='Run with debug logs'
    )
    args = parser.parse_args()
    if args.debug:
        mcs.LoggingConfig.init_logging(mcs.LoggingConfig.get_dev_logging_config())
    main(args.scene_file)

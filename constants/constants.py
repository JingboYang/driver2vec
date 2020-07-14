import os
import sys

from pathlib import Path

# Paths to various places

############# Constants for GCP bucket storage ##################
GC_HOME = Path(os.environ['HOME'])
SOURCE_PATH = GC_HOME / 'cs341_driver2vec'

PROJECT_ID = 
GC_BUCKET = 
CREDENTIAL_PATH = 

EXP_STORAGE = 'new_experiments'
DATA_STORAGE = 'driver_data'

TEMP_FOLDER = GC_HOME / 'temp_store'
TEMP_FOLDER.mkdir(exist_ok=True)

############### Caching generated masks ##########################

LOAD_MASK_FROM_FILE = True
LOCAL_MASK_PATH = GC_HOME / 'temp_store' / 'local_mask.pickle'

###################################################################

JUPYTER_PORT = 8964

# Things related to data
OVERLAP_PERCENT = 0.5       # Cannot overlap more than 50%

DATA_LENGTHS = {'cl': {'user_1594_scenario_3_repeat_0_opti': 68371,
                       'user_1636_scenario_0_repeat_0_opti': 83791,
                       'user_1642_scenario_0_repeat_1_opti': 94722,
                       'user_1727_scenario_2_repeat_0_opti': 43060,
                       'user_1772_scenario_1_repeat_0_opti': 92543,
                       'user_1800_scenario_0_repeat_0_opti': 107350,
                       'user_7974_scenario_1_repeat_0_opti': 90805,
                       'user_8161_scenario_1_repeat_1_opti': 77968,
                       'user_8164_scenario_1_repeat_0_opti': 81911},
                'nc': {'user_1634_scenario_0_repeat_0_opti': 105191,
                       'user_1710_scenario_1_repeat_0_opti': 92945,
                       'user_1830_scenario_0_repeat_0_opti': 120567,
                       'user_1929_scenario_1_repeat_0_opti': 93054,
                       'user_7901_scenario_1_repeat_0_opti': 85763,
                       'user_8112_scenario_1_repeat_0_opti': 106363}}

USEFULL_COLS = ['ACCELERATION', 'ACCELERATION_PEDAL', 'ACCELERATION_Y',
                'ACCELERATION_Z', 'BRAKE_PEDAL', 'CLUTCH_PEDAL',
                'DISTANCE', 'FOG', 'FOG_LIGHTS',
                'FRONT_WIPERS', 'GEARBOX', 'HEADING',
                'HEAD_LIGHTS', 'HORN', 'PITCH',
                'REAR_WIPERS', 'ROAD_SLOPE', 'ROLL',
                'SPEED', 'SPEED_Y', 'SPEED_Z',
                'STEERING_WHEEL', 'WHEEL_BASE']
NUM_SELECTED_COLS = len(USEFULL_COLS)

USELESS_COLS = ['ACCELERATION_PEDAL', 'ACCELERATION_Y',
                'ACCELERATION_Z', 'BRAKE_PEDAL', 'CLUTCH_PEDAL',
                'DISTANCE', 'FOG', 'FOG_LIGHTS',
                'FRONT_WIPERS', 'GEARBOX', 'HEADING',
                'HEAD_LIGHTS', 'HORN', 'PITCH',
                'REAR_WIPERS', 'ROAD_SLOPE', 'ROLL',
                'SPEED_Y', 'SPEED_Z',
                'WHEEL_BASE']

LABEL_TO_INDEX = {
    '1634': 0,
    '1830': 1,
    '7901': 2,
    '8112': 3,
    '1710': 4,
    '1929': 5,
    '8161': 6,
    '7974': 7,
    '1642': 8,
    '1727': 9,
    '8164': 10,
    '1594': 11,
    '1636': 12,
    '1800': 13,
    '1772': 14,
    '1583': 15,
    '1606': 16,
    '1632': 17,
    '1673': 18,
    '1709': 19,
    '1724': 20,
    '1735': 21,
    '1774': 22,
    '1784': 23,
    '1810': 24,
    '1825': 25,
    '1833': 26,
    '1835': 27,
    '1857': 28,
    '1888': 29,
    '1896': 30,
    '1897': 31,
    '1928': 32,
    '1933': 33,
    '1936': 34,
    '7907': 35,
    '7909': 36,
    '7916': 37,
    '8033': 38,
    '8038': 39,
    '8093': 40,
    '8111': 41,
    '8149': 42,
    '8166': 43,
    '8174': 44,
    '8200': 45,
    '8206': 46,
    '8210': 47,
    '8215': 48,
    '8218': 49,
    '8224': 50 
}


INDEX_TO_COLLISION = [
 0,
 0,
 1,
 0,
 0,
 0,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 0,
 0,
 1,
 0,
 1,
 1,
 1,
 0,
 1,
 0,
 0,
 1,
 0,
 1,
 1,
 0,
 1,
 0,
 0,
 0,
 1,
 0,
 1,
 0,
 1,
 0,
 0,
 1,
 0,
 0,
 1,
 1,
 1,
 1]

NUM_DRIVERS = len(LABEL_TO_INDEX)

USEFUL_COLS2=["ACCELERATION", "ACCELERATION_PEDAL", "ACCELERATION_Y",
              "ACCELERATION_Z", "BRAKE_PEDAL", "CLUTCH_PEDAL", "CURVE_RADIUS",
              "DISTANCE", "DISTANCE_TO_NEXT_INTERSECTION",
              "DISTANCE_TO_NEXT_STOP_SIGNAL",
              "DISTANCE_TO_NEXT_TRAFFIC_LIGHT_SIGNAL",
              "DISTANCE_TO_NEXT_VEHICLE",
              "DISTANCE_TO_NEXT_YIELD_SIGNAL",
              "FAST_LANE", "FOG", "FOG_LIGHTS", "FRONT_WIPERS", "GEARBOX",
              "HEADING", "HEAD_LIGHTS", "HORN", "INDICATORS",
              "INDICATORS_ON_INTERSECTION",
              "LANE", "LANE_LATERAL_SHIFT_CENTER", "LANE_LATERAL_SHIFT_LEFT",
              "LANE_LATERAL_SHIFT_RIGHT", "LANE_WIDTH", "RAIN",
              "REAR_WIPERS", "ROAD_ANGLE", "SNOW", "SPEED", 
              "SPEED_LIMIT", "SPEED_NEXT_VEHICLE", "SPEED_Y",
              "SPEED_Z", "STEERING_WHEEL"]

# Map group to list of columns
COLUMN_GROUPS = {
                "acceleration":["ACCELERATION",
                                "ACCELERATION_PEDAL",
                                "ACCELERATION_Y",
                                "ACCELERATION_Z"],
                "speed":["SPEED",
                        "SPEED_NEXT_VEHICLE",
                        "SPEED_Y",
                        "SPEED_Z"],
                "distance":["DISTANCE",
                            "DISTANCE_TO_NEXT_INTERSECTION",
                            "DISTANCE_TO_NEXT_STOP_SIGNAL",
                            "DISTANCE_TO_NEXT_TRAFFIC_LIGHT_SIGNAL",
                            "DISTANCE_TO_NEXT_VEHICLE",
                            "DISTANCE_TO_NEXT_YIELD_SIGNAL"],
                "pedal":["BRAKE_PEDAL","CLUTCH_PEDAL"],
                "lane":["LANE",
                        "LANE_LATERAL_SHIFT_CENTER",
                        "LANE_LATERAL_SHIFT_LEFT",
                        "LANE_LATERAL_SHIFT_RIGHT",
                        "LANE_WIDTH",
                        "FAST_LANE"],
                "fog":["FOG","FOG_LIGHTS"],
                "rainsnow":["RAIN","REAR_WIPERS","FRONT_WIPERS","SNOW"],
                "angle":["CURVE_RADIUS","ROAD_ANGLE","STEERING_WHEEL"],
                "indicator":["INDICATORS","INDICATORS_ON_INTERSECTION"],
                "headlight":["HEAD_LIGHTS"],
                # "horn":["HORN"],
                "gearbox":["GEARBOX"]
                        }

# List of 
COLUMN_SELECTION_SPECS = ["acceleration", "speed", "distance", "pedal",
                          "lane", "fog","rainsnow","angle",
                          "indicator","headlight","horn","gearbox"]

# dictionary of group key and columns used (everything else that is not in that group)
COLUMN_SELECTION_OPTIONS = {
    "no_acceleration":  list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['acceleration']))),
    "no_speed":         list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['speed']))),
    "no_distance":      list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['distance']))),
    "no_pedal":         list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['pedal']))),
    "no_lane":          list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['lane']))),
    "no_fog":           list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['fog']))),
    "no_rainsnow":      list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['rainsnow']))),
    "no_angle":         list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['angle']))),
    "no_indicator":     list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['indicator']))),
    "no_headlight":     list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['headlight']))),
    # "no_horn":          list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['horn']))),
    "no_gearbox":       list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['gearbox']))),
    "no_fog_rainsnow":  list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['fog']+COLUMN_GROUPS['rainsnow']))),
    "no_fog_rainsnow_headlight":  list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['fog']+COLUMN_GROUPS['rainsnow']+COLUMN_GROUPS['headlight']))),
    "everything":       list(set(USEFUL_COLS2)),
    "no_weather_acceleration":  list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['acceleration']+COLUMN_GROUPS['fog']+COLUMN_GROUPS['rainsnow']+COLUMN_GROUPS['headlight']))),
    "no_weather_speed":         list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['speed']+COLUMN_GROUPS['fog']+COLUMN_GROUPS['rainsnow']+COLUMN_GROUPS['headlight']))),
    "no_weather_distance":      list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['distance']+COLUMN_GROUPS['fog']+COLUMN_GROUPS['rainsnow']+COLUMN_GROUPS['headlight']))),
    "no_weather_pedal":         list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['pedal']+COLUMN_GROUPS['fog']+COLUMN_GROUPS['rainsnow']+COLUMN_GROUPS['headlight']))),
    "no_weather_lane":          list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['lane']+COLUMN_GROUPS['fog']+COLUMN_GROUPS['rainsnow']+COLUMN_GROUPS['headlight']))),
    "no_weather_angle":         list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['angle']+COLUMN_GROUPS['fog']+COLUMN_GROUPS['rainsnow']+COLUMN_GROUPS['headlight']))),
    "no_weather_indicator":     list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['indicator']+COLUMN_GROUPS['fog']+COLUMN_GROUPS['rainsnow']+COLUMN_GROUPS['headlight']))),
    # "no_horn":          list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['horn']))),
    "no_weather_gearbox":       list(set(USEFUL_COLS2).difference(set(COLUMN_GROUPS['gearbox']+COLUMN_GROUPS['fog']+COLUMN_GROUPS['rainsnow']+COLUMN_GROUPS['headlight']))),
    }

MAX_P_WAYS = 5

# Not too sure why the final layer even works ???
triplet_train_metrics = [('one_hot_accuracy', ()),
                         ('confusion_matrix', ()),
                         ('triplet_accuracy', ()),
                         ('triplet_ratio', ()),
                         ('triplet_diff_weight_ratio', ()),
                         ('tsne', ()),
                         ('tsne_collisions', ())]

triplet_simple_eval_metrics = [('one_hot_accuracy', ()),
                               ('confusion_matrix', ()),
                               ('triplet_accuracy', ()),
                               ('triplet_ratio', ()),
                               ('triplet_diff_weight_ratio', ()),
                               ('tsne', ()),
                               ('tsne_collisions', ())]
triplet_simple_eval_metrics.extend([('p_way_accuracy', (p, ))
                                    for p in range(2, MAX_P_WAYS + 1, 1)])
#triplet_simple_eval_metrics.extend([('area_accuracy', (p, ))
#                                    for p in [-1, 1, 8, 9]])

triplet_lgbm_eval_metrics = [('one_hot_accuracy', ()),
                             ('confusion_matrix', ()),
                             # Only need to do these once
                             # ('triplet_accuracy', ()),
                             # ('triplet_ratio', ()),
                             # ('triplet_diff_weight_ratio', ()),
                             # ('tsne', ()),
                             # ('tsne_collisions', ())
                             ]
#triplet_lgbm_eval_metrics.extend([('per_driver_f1', (i, ))
#                                  for i in range(NUM_DRIVERS)]) 
triplet_lgbm_eval_metrics.extend([('p_way_accuracy', (p, ))
                                  for p in range(2, MAX_P_WAYS + 1, 1)])
#triplet_lgbm_eval_metrics.extend([('area_accuracy', (p, ))
#                                  for p in [-1, 1, 8, 9]])

same_area_eval_metrics = [('triplet_accuracy', ()),
                          ('triplet_ratio', ()),
                          ('triplet_diff_weight_ratio', ())
                          ]
same_area_eval_metrics.extend([('area_accuracy', (a, ))
                               for a in [-1, 1, 8, 9]])

# MUST match name of datasets
# LightGBM not available during training because LightGBM training
# is not additive. Need to retrain with additional data for better trees.
TRIPLET_EVAL_METRICS = {'train': {'train': triplet_train_metrics},
                        'eval': {'eval_simple': (triplet_simple_eval_metrics,
                                                 'simple_predict'),
                                 'eval_lgbm': (triplet_lgbm_eval_metrics,
                                               'lgbm_predict'),
                                 },
                        'test': {'test_simple': (triplet_simple_eval_metrics,
                                                 'simple_predict'),
                                 'test_lgbm': (triplet_lgbm_eval_metrics,
                                               'lgbm_predict')}
                        }

DATASET_EVAL_METRICS = {'divided_splitted_normalized': TRIPLET_EVAL_METRICS,
                        'divided_splitted': TRIPLET_EVAL_METRICS,
                        'area_2_divided_splitted': TRIPLET_EVAL_METRICS,
                        'raw_normalized': TRIPLET_EVAL_METRICS}
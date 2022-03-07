ROOT_STATS_DIR = './experiment_data'

target_id_map = {'O': 0,
                'B-Lead': 1,
                'I-Lead': 2,
                'B-Position': 3,
                'I-Position': 4,
                'B-Claim': 5,
                'I-Claim': 6,
                'B-Counterclaim': 7,
                'I-Counterclaim': 8,
                'B-Rebuttal': 9,
                'I-Rebuttal': 10,
                'B-Evidence': 11,
                'I-Evidence': 12,
                'B-Concluding Statement': 13,
                'I-Concluding Statement': 14}


id_target_map = {v: k for k, v in target_id_map.items()}
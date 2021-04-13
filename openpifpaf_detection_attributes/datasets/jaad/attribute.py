from .. import attribute


class JaadType(attribute.ObjectType):
    """Object types for JAAD."""
    PEDESTRIAN = ()


JAAD_ATTRIBUTE_METAS = {
    JaadType.PEDESTRIAN: [
        # Detection
        {'attribute': 'confidence',          'group': 'detection',  'only_on_instance': False, 'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1},
        {'attribute': 'center',              'group': 'detection',  'only_on_instance': True,  'is_classification': False, 'is_scalar': False, 'is_spatial': True,  'n_channels': 2, 'std': [2.7, 5.9]},
        {'attribute': 'height',              'group': 'detection',  'only_on_instance': True,  'is_classification': False, 'is_scalar': True,  'is_spatial': True,  'n_channels': 1, 'default': 17.5, 'mean': 17.5, 'std': 8.9},
        {'attribute': 'width',               'group': 'detection',  'only_on_instance': True,  'is_classification': False, 'is_scalar': True,  'is_spatial': True,  'n_channels': 1, 'default': 7.7,  'mean': 7.7,  'std': 4.4},
        # Intention
        {'attribute': 'will_cross',          'group': 'intention',  'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'time_to_crossing',    'group': 'intention',  'only_on_instance': True,  'is_classification': False, 'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': -2.4, 'mean': -2.4, 'std': 2.8},
        # Behavior
        {'attribute': 'is_crossing',         'group': 'behavior',   'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'look',                'group': 'behavior',   'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'walk',                'group': 'behavior',   'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 1},
        {'attribute': 'motion_direction',    'group': 'behavior',   'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 2, 'default': 0, 'labels': {0: 'lateral', 1: 'longitudinal'}},
        {'attribute': 'pose_back',           'group': 'behavior',   'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'pose_front',          'group': 'behavior',   'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'pose_left',           'group': 'behavior',   'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'pose_right',          'group': 'behavior',   'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'group_size',          'group': 'behavior',   'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 4, 'default': 1, 'labels': {0: '1', 1: '2', 2: '3', 3: '4+'}},
        {'attribute': 'reaction',            'group': 'behavior',   'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 4, 'default': 0, 'labels': {0: 'none', 1: 'clear_path', 2: 'speed_up', 3: 'slow_down'}},
        # Appearance
        {'attribute': 'gender',              'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 2, 'default': 0, 'labels': {0: 'female', 1: 'male'}},
        {'attribute': 'backpack',            'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'bag_elbow',           'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'bag_hand',            'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'bag_left_side',       'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'bag_right_side',      'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'bag_shoulder',        'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'cap',                 'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'clothes_below_knee',  'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'clothes_lower_dark',  'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 1},
        {'attribute': 'clothes_upper_dark',  'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 1},
        {'attribute': 'clothes_lower_light', 'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'clothes_upper_light', 'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'hood',                'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'object',              'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'phone',               'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'stroller_cart',       'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'sunglasses',          'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        {'attribute': 'age',                 'group': 'appearance', 'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 3, 'default': 1, 'labels': {0: 'child/young', 1: 'adult', 2: 'senior'}},
        # Not used
        #{'attribute': 'baby',                'group': 'notused',    'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        #{'attribute': 'bicycle_motorcycle',  'group': 'notused',    'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        #{'attribute': 'hand_gesture',        'group': 'notused',    'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        #{'attribute': 'nod',                 'group': 'notused',    'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
        #{'attribute': 'umbrella',            'group': 'notused',    'only_on_instance': True,  'is_classification': True,  'is_scalar': True,  'is_spatial': False, 'n_channels': 1, 'default': 0},
    ],
}

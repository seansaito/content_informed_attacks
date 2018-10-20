import numpy as np

from constants import ATTACKS

import cleverhans.attacks as attacks

attack_name_prefix = '{targeted_prefix}_{attack_name}_model-{model}'

attack_to_prefix_template = {
    ATTACKS.LBFGS: '_binary_search_steps-{binary_search_steps}_max_iterations-{max_iterations}_initial_const-{initial_const}',
    ATTACKS.FGSM: '_eps-{eps}',
    ATTACKS.BIM: '_nb_iter-{nb_iter}_eps-{eps}_eps_iter-{eps_iter}',
    ATTACKS.MIM: '_nb_iter-{nb_iter}_eps-{eps}_eps_iter-{eps_iter}',
    ATTACKS.VIRTUAL_ATTACK: '_num_iterations-{num_iterations}_xi-{xi}_eps-{eps}',
    ATTACKS.DEEP_FOOL: '_nb_candidate-{nb_candidate}_max_iter-{max_iter}_overshoot-{overshoot}',
    ATTACKS.MADRY: '_nb_iter-{nb_iter}_eps-{eps}_eps_iter-{eps_iter}',
    ATTACKS.CARLINI_WAGNER: '_binary_search_steps-{binary_search_steps}_learning_rate-{learning_rate}_initial_const-{initial_const}_max_iterations-{max_iterations}'
}

attack_name_to_class = {
    ATTACKS.CARLINI_WAGNER: attacks.CarliniWagnerL2,
    ATTACKS.FGSM: attacks.FastGradientMethod,
    ATTACKS.MIM: attacks.MomentumIterativeMethod,
    ATTACKS.BIM: attacks.BasicIterativeMethod,
    ATTACKS.SALIENCY_MAP: attacks.SaliencyMapMethod,
    ATTACKS.VIRTUAL_ATTACK: attacks.VirtualAdversarialMethod,
    ATTACKS.ELASTIC_NET: attacks.ElasticNetMethod,
    ATTACKS.DEEP_FOOL: attacks.DeepFool,
    ATTACKS.LBFGS: attacks.LBFGS,
    ATTACKS.MADRY: attacks.MadryEtAl,
    ATTACKS.FAST_FEATURES: attacks.FastFeatureAdversaries
}

attack_name_to_params = {
    ATTACKS.CARLINI_WAGNER: {
        'binary_search_steps': [1, 5], # 5 is better
        'max_iterations': 1000, # 1000 is best
        'learning_rate': 0.01,
        'batch_size': 10,
        'initial_const': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    },
    ATTACKS.CARLINI_WAGNER + '_quick': {
        'binary_search_steps': 5,  # 5 is better
        'max_iterations': 1000,  # 1000 is best
        'learning_rate': 0.1,
        'batch_size': 50,
        'initial_const': 100000.0
    },
    ATTACKS.MIM: {
        'eps': list(np.arange(0.0, 0.31, 0.01)),
        'eps_iter': 0.06,
        'nb_iter': 10  # should be 10
    },
    ATTACKS.MIM + '_quick': {
        'eps': 30.0,
        'eps_iter': 1.0,
        'nb_iter': 10  # should be 10
    }
}

attack_name_to_configurable_params = {
    ATTACKS.CARLINI_WAGNER: ['initial_const', 'binary_search_steps'],
    ATTACKS.FGSM: 'eps',
    ATTACKS.BIM: 'eps',
    ATTACKS.MIM: 'eps',
    ATTACKS.VIRTUAL_ATTACK: 'eps',
    ATTACKS.ELASTIC_NET: 'beta',
    ATTACKS.DEEP_FOOL: 'max_iter',
    ATTACKS.LBFGS: 'initial_const',
    ATTACKS.MADRY: 'eps',
}

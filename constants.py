class LOSSES:
    CE = 'cross_entropy'
    MSE = 'mean_squared_error'

class ACTIVATIONS:
    SOFTMAX = 'Softmax'

class ATTACKS:
    CARLINI_WAGNER = 'carlini_wagner'
    FGSM = 'fgsm'
    MIM = 'mim'
    BIM = 'bim'
    SALIENCY_MAP = 'saliency_map'
    VIRTUAL_ATTACK = 'vat'
    ELASTIC_NET = 'elastic_net'
    DEEP_FOOL = 'deep_fool'
    LBFGS = 'lbfgs'
    MADRY = 'madry_et_al'
    FAST_FEATURES = 'fast_features_adversaries'

class PATHS:
    ADVERSARIAL_EXAMPLES = 'adversarial_examples'
    ILSVRC_PATH = '/experiments/ImageNet/ILSVRC'
    IMAGENET_TRAIN_LMDB = '/experiments/ImageNet/imagenet_train_lmdb'
    IMAGENET_VALID_LMDB = '/experiments/ImageNet/imagenet_valid_lmdb'
    DATA_PATH = '/experiments/ImageNet/ILSVRC/Data/CLS-LOC'
    MODEL_SAVE_PATH = 'saves'

class TARGET_TYPES:
    ONEHOT_CE = 'onehot_ce'
    ONEHOT_MSE = 'onehot_mse'
    ONEHOT_SIGMOID = 'onehot_sigmoid'
    RANDOM = 'random'

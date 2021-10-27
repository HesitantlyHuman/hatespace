from ray import tune

config = {
    'device' : 'cuda:0',
    'losses' : {
        'distribution' : {
            'weight' : 136.1018,
            'type' : 'sinkhorn',
            'p' : 2,
            'blur' : 0.05,
            'alpha' : 1.0
        },
        'class' : {
            'weight' : 1.0443,
            'bias' : 1.0,
            'threshold' : 0.0
        },
        'reconstruction' : {
            'weight' : 202.9770
        }
    },
    'training' : {
        'max_epochs' : 25,
        'batch_size' : tune.randint(16, 128)
    },
    'dataset' : {
        'directory' : 'iron_march_201911',
        'context' : tune.choice([True, False]),
        'side_information' : {
            'file_paths' : [
                'side_information/hate_words/processed_side_information.csv'
            ]
        }
    },
    'latent_space' : {
        'noise' : {
            'std' : tune.uniform(0.1, 0.7)
        }
    },
    'adam' : {
        'max_learning_rate' : tune.loguniform(1e-3, 1e-2),
        'weight_decay' : tune.uniform(0.01, 0.04),
        'betas' : {
            'zero' : tune.loguniform(0.8, 0.9),
            'one' : tune.loguniform(0.95, 0.995)
        }
    },
    'model' : {
        'latent_dims' : 16,
        'max_dropout' : tune.uniform(0.1, 0.5),
        'encoder' : {
            'depth' : tune.randint(3, 15),
            'bias' :  tune.loguniform(0.1, 10)
        },
        'decoder' : {
            'depth' : tune.randint(3, 15),
            'bias' :  tune.loguniform(0.1, 10)
        },
        'softmax' : True
    }
}
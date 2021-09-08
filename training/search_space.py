from ray import tune

config = {
    'device' : 'cuda:0',
    'losses' : {
        'distribution' : {
            'weight' : 0.0,
            'type' : 'sinkhorn',
            'p' : 2,
            'blur' : 0.05,
            'alpha' : 1.0
        },
        'class' : {
            'weight' : 0.0,
            'bias' : 1.0,
            'threshold' : 0.0
        },
        'reconstruction' : {
            'weight' : 1.0
        }
    },
    'training' : {
        'max_epochs' : 25,
        'batch_size' : tune.randint(4, 128)
    },
    'dataset' : {
        'directory' : 'iron_march_201911',
        'context' : tune.choice([True, False]),
        'side_information' : {
            'file_paths' : [
                'side_information\hate_words\processed_side_information.csv'
            ]
        }
    },
    'latent_space' : {
        'noise' : {
            'std' : tune.uniform(0.1, 0.7)
        }
    },
    'adam' : {
        'max_learning_rate' : tune.loguniform(3e-4, 3e-2),
        'weight_decay' : tune.loguniform(0.01, 0.05),
        'betas' : {
            'zero' : tune.loguniform(0.8, 0.9),
            'one' : tune.loguniform(0.95, 0.995)
        }
    },
    'model' : {
        'latent_dims' : 32,
        'max_dropout' : tune.loguniform(0.05, 0.5),
        'encoder' : {
            'depth' : tune.randint(3, 18),
            'bias' :  tune.loguniform(0.1, 10)
        },
        'decoder' : {
            'depth' : tune.randint(3, 18),
            'bias' :  tune.loguniform(0.1, 10)
        },
        'softmax' : True
    }
}
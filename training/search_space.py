from ray import tune

config = {
    'device' : 'cuda:0',
    'losses' : {
        'distribution' : {
            'weight' : 1.0,
            'type' : 'sinkhorn',
            'p' : 2,
            'blur' : 0.05,
            'alpha' : 1.0
        },
        'class' : {
            'weight' : 0.05,
            'bias' : 1.0,
            'threshold' : 0.5
        },
        'reconstruction' : {
            'weight' : 10.0
        }
    },
    'training' : {
        'max_epochs' : 100,
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
            'std' : tune.uniform(0.0, 0.5)
        }
    },
    'adam' : {
        'learning_rate' : tune.loguniform(1e-7, 1e-2),
        'betas' : {
            'zero' : tune.loguniform(0.8, 0.99),
            'one' : tune.loguniform(0.9, 0.99999)
        }
    },
    'model' : {
        'latent_dims' : 16,
        'encoder' : {
            'depth' : tune.randint(3, 18),
            'bias' :  tune.loguniform(0.05, 20)
        },
        'decoder' : {
            'depth' : tune.randint(3, 18),
            'bias' :  tune.loguniform(0.05, 20)
        },
        'softmax' : True
    }
}
from ray import tune

config = {
    'training' : {
        'max_epochs' : 100,
        'batch_size' : tune.randint(2, 512)
    }
}

'''
config = {
    'losses' : {
        'distribution' : {
            'weight' : 1.0,
            'metric' : 'sinkhorn',
            'p' : 2,
            'blur' : 0.05,
            'alpha' : 1.0
        },
        'class' : {
            'weight' : 0.05,
            'bias' : 1.5,
            'threshold' : 0.5
        },
        'reconstruction' : {
            'weight' : 10.0
        }
    },
    'training' : {
        'max_epochs' : 100,
        'batch_size' : tune.randint(2, 512)
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
        'learning_rate' : tune.loguniform(1e-7, 1e-1),
        'betas' : [
            tune.loguniform(0.8, 0.99),
            tune.loguniform(0.9, 0.99999)
        ]
    },
    'model' : {
        'latent_dims' : 16,
        'encoder' : {
            'depth' : tune.randint(1, 15),
            'bias' :  tune.loguniform(0.1, 10)
        },
        'decoder' : {
            'depth' : tune.randint(1, 20),
            'bias' :  tune.loguniform(0.1, 10)
        },
        'softmax' : tune.choice([True, False])
    }
}
'''
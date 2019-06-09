#!/usr/bin/env bash

python3 run_experiment.py --datapath data/mokoron --dataset-name mokoron --model-name YoonKim

# OLD:

# python3 noise_experiment_tweets.py --datapath data/mokoron --dataset-name mokoron --model-name CharCNN --comment _main_exp
# echo -e 'FastText model'
# python3 noise_experiment_tweets.py --datapath data/mokoron --dataset-name mokoron --model-name FastText --embeddings-path /data/embeddings/wiki.ru.bin --comment _main_exp
# echo -e 'YoonKim model'
# python3 noise_experiment_tweets.py --datapath data/mokoron --dataset-name mokoron --model-name YoonKim --comment _main_exp
# echo -e 'AttentionedYoonKim model'
# python3 noise_experiment_tweets.py --datapath data/mokoron --dataset-name mokoron --model-name AttentionedYoonKim --comment _main_exp

# echo -e '\n\nAirline tweets experiment is started\n\n'
# echo -e 'CharCNN model'
# python3 noise_experiment_tweets.py --datapath data/airline_tweets --dataset-name airline-tweets --model-name CharCNN --comment _main_exp
# echo -e 'FastText model'
# python3 noise_experiment_tweets.py --datapath data/airline_tweets --dataset-name airline-tweets --model-name FastText --comment _main_exp
# echo -e 'YoonKim model'
# python3 noise_experiment_tweets.py --datapath data/airline_tweets --dataset-name airline-tweets --model-name YoonKim --comment _main_exp
# echo -e 'AttentionedYoonKim model'
# python3 noise_experiment_tweets.py --datapath data/airline_tweets --dataset-name airline-tweets --model-name AttentionedYoonKim --comment _main_exp

# echo -e '\n\nRusentiment\n\n'
# python3 noise_experiment_tweets.py --datapath /data/classification/rusentiment/preselected --dataset-name rusentiment --model-name CharCNN --comment _main_exp
# echo -e 'FastText model'
# python3 noise_experiment_tweets.py --datapath /data/classification/rusentiment/preselected --dataset-name rusentiment --model-name FastText --embeddings-path /data/embeddings/wiki.ru.bin --comment _main_exp
# echo -e 'YoonKim model'
# python3 noise_experiment_tweets.py --datapath /data/classification/rusentiment/preselected --dataset-name rusentiment --model-name YoonKim --comment _main_exp
# echo -e 'CharCNN model'
# python3 noise_experiment_tweets.py --datapath /data/classification/rusentiment/preselected --dataset-name rusentiment --model-name CharCNN --comment _main_exp --noise-level 0.0 --original-train
# echo -e 'FastText model'
# python3 noise_experiment_tweets.py --datapath /data/classification/rusentiment/preselected --dataset-name rusentiment --model-name FastText --embeddings-path /data/embeddings/wiki.ru.bin --comment _main_exp --noise-level 0.0 --original-train
# echo -e 'YoonKim model'
# python3 noise_experiment_tweets.py --datapath /data/classification/rusentiment/preselected --dataset-name rusentiment --model-name YoonKim --comment _main_exp --noise-level 0.0 --original-train
# echo -e 'ELMo model'
# python3 noise_experiment_tweets.py --datapath /data/classification/rusentiment/preselected_spellchecked --dataset-name rusentiment --model-name ELMo --comment _main_exp_elmo

# echo -e '\n\nIMDB experiment is started\n\n'
# echo -e 'CharCNN model'
# python3 noise_experiment_imdb.py --model-name CharCNN --comment _main_exp
# echo -e 'FastText model'
# python3 noise_experiment_imdb.py --model-name FastText --comment _main_exp
# echo -e 'YoonKim model'
# python3 noise_experiment_imdb.py --model-name YoonKim --comment _main_exp
# echo -e 'AttentionedYoonKim model'
# python3 noise_experiment_imdb.py --model-name AttentionedYoonKim --comment _main_exp

# echo -e '\n\nSentiRuEval\n\n'
# python3 noise_experiment_tweets.py --datapath /data/classification/SentiRuEval_data/all_data --dataset-name sentirueval --model-name CharCNN --comment _main_exp
# echo -e 'FastText model'
# python3 noise_experiment_tweets.py --datapath /data/classification/SentiRuEval_data/all_data --dataset-name sentirueval --model-name FastText --embeddings-path /data/embeddings/wiki.ru.bin --comment _main_exp
# echo -e 'YoonKim model'
# python3 noise_experiment_tweets.py --datapath /data/classification/SentiRuEval_data/all_data --dataset-name sentirueval --model-name YoonKim --comment _main_exp

# python3 noise_experiment_tweets.py --datapath /data/classification/SentiRuEval_data/all_data --dataset-name sentirueval --model-name CharCNN --comment _main_exp --noise-level 0.0 --original-train
# python3 noise_experiment_tweets.py --datapath /data/classification/SentiRuEval_data/all_data --dataset-name sentirueval --model-name FastText --embeddings-path /data/embeddings/wiki.ru.bin --comment _main_exp --noise-level 0.0 --original-train
# python3 noise_experiment_tweets.py --datapath /data/classification/SentiRuEval_data/all_data --dataset-name sentirueval --model-name YoonKim --comment _main_exp --noise-level 0.0 --original-train


# python3 noise_experiment_tweets.py --datapath data/airline_tweets --dataset-name airline-tweets --model-name CharCNN --comment _main_exp --noise-level 0.0 --original-train
# python3 noise_experiment_tweets.py --datapath data/airline_tweets --dataset-name airline-tweets --model-name FastText --embeddings-path /data/embeddings/wiki.ru.bin --comment _main_exp --noise-level 0.0 --original-train
# python3 noise_experiment_tweets.py --datapath data/airline_tweets --dataset-name airline-tweets --model-name YoonKim --comment _main_exp --noise-level 0.0 --original-train

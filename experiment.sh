#!/usr/bin/env bash

echo -e '\n\nIMDB experiment is started\n\n'
echo -e 'CharCNN model'
python3 experiment_imdb.py --model-name CharCNN --comment _main_exp
echo -e 'FastText model'
python3 experiment_imdb.py --model-name FastText --comment _main_exp
echo -e 'YoonKim model'
python3 experiment_imdb.py --model-name YoonKim --comment _main_exp
echo -e 'AttentionedYoonKim model'
python3 experiment_imdb.py --model-name AttentionedYoonKim --comment _main_exp

echo -e '\n\nMokoron experiment is started\n\n'
echo -e 'CharCNN model'
python3 experiment_tweets.py --dataset-name mokoron--model-name CharCNN --comment _main_exp
echo -e 'FastText model'
python3 experiment_tweets.py --dataset-name mokoron --model-name FastText --comment _main_exp
echo -e 'YoonKim model'
python3 experiment_tweets.py --dataset-name mokoron --model-name YoonKim --comment _main_exp
echo -e 'AttentionedYoonKim model'
python3 experiment_tweets.py --dataset-name mokoron --model-name AttentionedYoonKim --comment _main_exp

echo -e '\n\nAirline tweets experiment is started\n\n'
echo -e 'CharCNN model'
python3 experiment_tweets.py --dataset-name airline-tweets --datapath data/airline_tweets --model-name CharCNN --comment _main_exp
echo -e 'FastText model'
python3 experiment_tweets.py --dataset-name airline-tweets --datapath data/airline_tweets --model-name FastText --comment _main_exp
echo -e 'YoonKim model'
python3 experiment_tweets.py --dataset-name airline-tweets --datapath data/airline_tweets --model-name YoonKim --comment _main_exp
echo -e 'AttentionedYoonKim model'
python3 experiment_tweets.py --dataset-name airline-tweets --datapath data/airline_tweets --model-name AttentionedYoonKim --comment _main_exp

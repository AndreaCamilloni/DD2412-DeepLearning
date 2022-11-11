# Script for training the model

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Self-Supervised Classification')
    parser.add_argument('--data_dir', default='data', type=str, help='Path to the data directory')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Backbone architecture')
    
    # Batch size, learning rate, epochs, parameters ...
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)

    # ...............................................

if __name__ == '__main__':
    main()



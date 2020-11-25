import os
import json
import torch
import pickle
import numpy as np
import argparse

from gradient_attack_utils import get_exact_matches
from dataset import Vocabulary, create_dataloader
from model import Code2Seq


def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--orig_data_path', action='store', dest='orig_data_path', help='Path to original data')
	parser.add_argument('--data_path', action='store', dest='data_path', help='Path to transformed dataset')
	parser.add_argument('--checkpoint', action='store', dest='checkpoint', help='Path to checkpoint')
	parser.add_argument('--batch_size', type=int, action='store', dest='batch_size')
	parser.add_argument('--split', action='store', dest='split')
	parser.add_argument('--vocab_path', action='store')

	args = parser.parse_args()
	return args

def create_datafile(data_path, exact_matches, split):

	new_data_path = os.path.join(data_path, 'small.{}.c2s'.format(split))
	lines = open(os.path.join(data_path, 'data.{}.c2s'.format(split)), 'r')
	new_file = open(new_data_path, 'w')
	for line in lines:
		if line.split()[0] in exact_matches:
			new_file.write(line)
	print("Saved exact matches.")

if __name__ == '__main__':

	args = parse_args()
	model = Code2Seq.load_from_checkpoint(checkpoint_path=args.checkpoint)
	data_loader, n_samples = create_dataloader(
		os.path.join(args.orig_data_path, args.split), model.hyperparams.max_context, False, False, args.batch_size, 1
	)
	vocab = pickle.load(open(args.vocab_path, 'rb'))
	label_to_id = vocab['label_to_id']
	id_to_label = {label_to_id[l]:l for l in label_to_id}

	li_exact_matches = get_exact_matches(data_loader, n_samples, model, id_to_label)
	print(li_exact_matches)
	create_datafile(args.data_path, li_exact_matches, args.split)






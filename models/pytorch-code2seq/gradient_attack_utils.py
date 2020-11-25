import re
import numpy as np
import torch
import random
import tqdm

import torchtext
from torch.autograd import Variable

def classify_tok(tok):
	PY_KEYWORDS = re.compile(
	  r'^(False|class|finally|is|return|None|continue|for|lambda|try|True|def|from|nonlocal|while|and|del|global|not|with|as|elif|if|or|yield|assert|else|import|pass|break|except|in|raise)$'
	)

	JAVA_KEYWORDS = re.compile(
	  r'^(abstract|assert|boolean|break|byte|case|catch|char|class|continue|default|do|double|else|enum|exports|extends|final|finally|float|for|if|implements|import|instanceof|int|interface|long|module|native|new|package|private|protected|public|requires|return|short|static|strictfp|super|switch|synchronized|this|throw|throws|transient|try|void|volatile|while)$'
	)

	NUMBER = re.compile(
	  r'^\d+(\.\d+)?$'
	)

	BRACKETS = re.compile(
	  r'^(\{|\(|\[|\]|\)|\})$'
	)

	OPERATORS = re.compile(
	  r'^(=|!=|<=|>=|<|>|\?|!|\*|\+|\*=|\+=|/|%|@|&|&&|\||\|\|)$'
	)

	PUNCTUATION = re.compile(
	  r'^(;|:|\.|,)$'
	)

	WORDS = re.compile(
	  r'^(\w+)$'
	)


	if PY_KEYWORDS.match(tok):
		return 'KEYWORD'
	elif JAVA_KEYWORDS.match(tok):
		return 'KEYWORD'
	elif NUMBER.match(tok):
		return 'NUMBER'
	elif BRACKETS.match(tok):
		return 'BRACKET'
	elif OPERATORS.match(tok):
		return 'OPERATOR'
	elif PUNCTUATION.match(tok):
		return 'PUNCTUATION'
	elif WORDS.match(tok):
		return 'WORDS'
	else:
		return 'OTHER'

def get_valid_token_mask(negation, id_to_token, exclude):
	mask_valid = []
	for i in range(len(id_to_token)):
		if negation:
			mask_valid.append(not valid_replacement(id_to_token[i], exclude=exclude))
		else:
			mask_valid.append(valid_replacement(id_to_token[i], exclude=exclude))
	return mask_valid

def valid_replacement(s, exclude=[]):
	return classify_tok(s)=='WORDS' and s not in exclude

def convert_to_onehot(inp, vocab_size, device):
	return torch.zeros(inp.size(0), inp.size(1), vocab_size, device=device).scatter_(2, inp.unsqueeze(2), 1.)

def get_all_replacement_toks(input_var, input_orig, token_to_id, replace_tokens):
	d_temp, site_map, status = {}, {}, False
	for repl_tok in replace_tokens:
		if (repl_tok not in token_to_id) or (token_to_id[repl_tok] not in input_var):
			continue
		repl_tok_idx = token_to_id[repl_tok]
		status = True
		mask = input_var==repl_tok_idx
		if repl_tok_idx in site_map:
			assert False
		site_map[repl_tok_idx] = mask
	return d_temp, site_map, status

def calculate_loss(use_cw_loss, loss_obj, decoder_outputs, other, target_variables):
	token_wise_loss_per_batch = None
	if use_cw_loss:
		loss, token_wise_loss_per_batch = loss_obj.get_loss(other['logits'], target_variables)
		l_scalar = loss
	else:
		loss_obj.reset()
		for step, step_output in enumerate(decoder_outputs):
			batch_size = target_variables.size(0)
			l = torch.nn.NLLLoss(reduction='none')(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1]).unsqueeze(dim=1)
			# dim of l: batch_sz x token_i of output
			if token_wise_loss_per_batch is None:
				token_wise_loss_per_batch = l
			else:
				token_wise_loss_per_batch = torch.cat((token_wise_loss_per_batch, l), 1)
			loss_obj.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])

		# dim of token_wise_loss_per_batch = batch_sz x 1 
		token_wise_loss_per_batch = torch.mean(token_wise_loss_per_batch, dim=1).detach().cpu().numpy()
		loss = loss_obj
		l_scalar = loss_obj.get_loss()                    
	
	return loss, l_scalar, token_wise_loss_per_batch

def pad_inputs(new_inputs, new_site_map_map, z_all_map, token_to_id, max_size, updated_lengths):
	"""
	Pad new inputs, site maps, and z maps to length max_size.
	"""

	res_inputs, res_site_map_map, res_z_all_map = [], {}, {}
	
	for i in range(len(new_inputs)):
		
		if len(new_inputs[i]) == max_size:
			res_inputs.append(new_inputs[i])
			if i in new_site_map_map:
				res_site_map_map[i] = new_site_map_map[i]
				res_z_all_map[i] = z_all_map[i]
		else:
			res_inputs.append(new_inputs[i]+[input_vocab.stoi['<pad>'] for j in range(max_size-len(new_inputs[i]))])
			if i in new_site_map_map:
				res_site_map_map[i] = {}
				for r in new_site_map_map[i]:
					res_site_map_map[i][r] = new_site_map_map[i][r]+[False for j in range(max_size-len(new_inputs[i]))]
			if i in z_all_map:
				res_z_all_map[i] = z_all_map[i] + [False for j in range(max_size-len(new_inputs[i]))]
				
	return res_inputs, res_site_map_map, res_z_all_map
		

def replace_toks_sample(input_var, input_var_len, z, site_map, site_map_lookup, best_replacements_sample, orig_replacements, token_to_id, id_to_token, field_name):

	"""
	input_var: (max_len, )
	site_map: maps sites to masks of length = max_len
	site_map_lookup: maps indices to site indices in input_vocab
	"""

	# # e.g. replace 'print ( @R_1@ )' with '@R_1@'
	# input_var_str = ' '.join([input_vocab.itos[t] for t in input_var])
	# # print('before', input_var_str)
	# for key in orig_replacements:
	# 	repl_tok = '@'+key.split('@')[1]+'@'
	# 	input_var_str = input_var_str.replace(key, repl_tok)
	# # print('after', input_var_str)
	# input_var = [input_vocab.stoi[t] for t in input_var_str.split(' ')]
			
	
	# find replacement tokens for sites
	toks_to_be_replaced = {}
	sites_to_fix = []
	for i in range(len(z)):
		repl_tok = id_to_token[site_map_lookup[i]]
		repl_tok_idx = site_map_lookup[i]
		
		for key in orig_replacements:
			if repl_tok in key:

				if orig_replacements[key][1] in ['transforms.InsertPrintStatements', 'transforms.AddDeadCode']:
					sites_to_fix.append(repl_tok_idx)

				if z[i] and repl_tok in best_replacements_sample:
					replaced_key = key.replace(repl_tok, best_replacements_sample[repl_tok])
					toks_to_be_replaced[repl_tok_idx] = [token_to_id[t] for t in replaced_key.split(' ')]
					
				else:
					toks_to_be_replaced[repl_tok_idx] = [token_to_id[t] for t in orig_replacements[key][0].split(' ')]

				break
	

	# update input (replace @R tokens)

	new_input = [] # list of lists
	updated_input = [] # list of indices
	for tok_idx in input_var:
		if tok_idx not in toks_to_be_replaced:
			new_input.append([tok_idx])   
			updated_input += [tok_idx]
		else:
			new_input.append(toks_to_be_replaced[tok_idx])
			updated_input += toks_to_be_replaced[tok_idx]


	# get new lengths
	updated_lengths = []
	start_idx = 0
	for i in input_var_len:
		end_idx = start_idx + i
		new_inp_chunk = new_input[start_idx : end_idx]
		updated_lengths.append(sum(len(new_inp_chunk[j]) for j in new_inp_chunk))



	# update site_map
	new_site_map = {}
	for r_idx in site_map_lookup:
		new_site_map[r_idx] = []
		for i in range(len(new_input)):
			if input_var[i] == r_idx:
				new_site_map[r_idx] += [True for j in range(len(new_input[i]))]
			else:
				new_site_map[r_idx] += [False for j in range(len(new_input[i]))]
		assert(len(new_site_map[r_idx]) == len(updated_input))

	# update z_map
	mask = np.array(np.array(updated_input)*[False]).astype(bool)
	for kk in range(len(site_map_lookup)):
		if not z[kk]:
			continue
		m = new_site_map[site_map_lookup[kk]]
		mask = np.array(m) | mask
	assert mask.shape[0] == len(updated_input)

	# print('input var', [input_vocab.itos[t] for t in input_var])
	# print('updated input', [input_vocab.itos[t] for t in updated_input])

	return updated_input, new_site_map, list(mask), max(updated_lengths), sites_to_fix, updated_lengths

def replace_toks_batch(input_vars, input_vars_lens, filenames, z_map, site_map_map, site_map_lookup_map, best_replacements_batch, field_name, token_to_id, id_to_token, orig_tok_map):

	"""
	inputs: (batch_size, max_len)
	indices: (max_len, )
	z_map: dict mapping sample idx to z (dim of z = num of sites in sample)
	site_map_map: dict mapping samples to site_maps
	site_map_lookup_map: dict mapping samples to site_map_lookup
	best_replacements_batch: dict mapping samples to the best replacements for certain sites

	Replaces z=1 sites with tokens from best_replacements_batch and
	z=0 sites with original tokens from orig_tok_map. If z=1 for some site but the site 
	is not in best_replacements (this happens in the first iteration), 
	it is also replaced with original tokens.
	"""

	new_inputs, new_site_map_map, z_all_map, sites_to_fix_map, new_updated_lengths = [], {}, {}, {}, []
	max_size = None

	for i in range(len(input_vars)):
		print(i)

		if i not in site_map_map:
			print(site_map_map)

			assert i not in z_map
			new_inp = input_vars[i]
			new_inp_max_size = max(input_vars_lens[i])
			if max_size is None or new_inp_max_size > max_size:
				max_size = new_inp_max_size
			continue

		z = z_map[i]
		site_map = site_map_map[i]
		site_map_lookup = site_map_lookup_map[i]
		input_var = input_vars[i]
		sample = filenames[i]
		input_var_len = input_vars_lens[i]
		
		if sample in best_replacements_batch:
			best_replacements_sample = best_replacements_batch[sample]
		else:
			best_replacements_sample = {}
		orig_replacements = orig_tok_map[sample]

		# update sample
		print('updating sample')
		new_inp, new_site_map, mask, new_max_length, sites_to_fix, new_inp_var_len = replace_toks_sample(
			input_var, input_var_len, z, site_map, site_map_lookup, best_replacements_sample, orig_replacements, token_to_id, id_to_token, field_name
			)

		new_inputs.append(new_inp)
		new_max_lengths.append(new_max_length)
		new_site_map_map[i] = new_site_map
		z_all_map[i] = mask
		sites_to_fix_map[i] = sites_to_fix
		new_updated_lengths[i] = new_inp_var_len
		if max_size is None or new_max_length > max_size:
			max_size = new_max_length

	# pad all samples in batch to max_size length
	# new_inputs, new_site_map_map, z_all_map = pad_inputs(new_inputs, new_site_map_map, z_all_map, token_to_id, max_size, new_updated_lengths)

	return new_inputs, new_site_map_map, z_all_map, new_updated_lengths, sites_to_fix_map, max_size

def modify_onehot(inputs_oho, site_map_map, sites_to_fix_map, device):

	for i in range(inputs_oho.shape[0]):

		if i in site_map_map:
			site_map = site_map_map[i]
			sites_to_fix = sites_to_fix_map[i]

			for site in sites_to_fix:
				mask = site_map[site]
				inputs_oho[i][mask] = torch.zeros(inputs_oho[i][mask].shape, requires_grad=True, device=device).half()

	return inputs_oho


def get_all_replacements(best_replacements, field_name, orig_tok_map, idx_to_fname, only_processed=False):
	"""
	This function creates a dictionary where optimized sites map to their best replacements
	and unoptimized ones map to their original tokens. This dictionary should be returned
	by apply_gradient_attack_v2 and should be used in replace_tokens.py
	"""

	all_replacements = {}
	avg_replaced, tot_replaced = 0, 0

	for idx in idx_to_fname:
		if only_processed and idx not in best_replacements:
			continue

		fname = idx_to_fname[idx]
		# add optimized site replacements
		if idx in best_replacements:
			all_replacements[idx] = {site:best_replacements[idx][site] for site in best_replacements[idx]}
			avg_replaced += len(best_replacements[idx])
		else:
			all_replacements[idx] = {}

		# find keys in orig_tok_map[fname] that don't contain optimized R sites
		valid_keys = []
		for key in orig_tok_map[fname]:
			valid = True
			for repl_tok in best_replacements[idx]:
				if repl_tok in key:
					valid = False
					break
			if valid:
				valid_keys.append(key)

		# add unoptimized site replacements
		to_add = {s:orig_tok_map[fname][s][0] for s in valid_keys}
		all_replacements[idx].update(to_add)
		tot_replaced += 1

	if tot_replaced == 0:
		avg_replaced = 0
	else:
		avg_replaced /= tot_replaced
	return all_replacements, avg_replaced

def remove_padding_and_flatten(inp_tokens, token_to_id):
	inp_tokens_flat, inp_tokens_lengths = [], []
	pad_idx = token_to_id['<PAD>']
	for sample in inp_tokens:
		flat_inp, inp_lens = [], []
		sample = sample.cpu().numpy()
		for col in range(sample.shape[1]):
			inp = sample[:,col].tolist()
			if pad_idx in inp:
				inp = inp[:inp.index(pad_idx)]
			flat_inp += inp
			inp_lens.append(len(inp))
		inp_tokens_flat.append(flat_inp)
		inp_tokens_lengths.append(inp_lens)
	return inp_tokens_flat, inp_tokens_lengths


def bisection(f,a,b,N):	
	# From https://www.math.ubc.ca/~pwalls/math-python/roots-optimization/bisection/	
	# '''Approximate solution of f(x)=0 on interval [a,b] by bisection method.	
	#	
	# Parameters	
	# ----------	
	# f : function	
	#     The function for which we are trying to approximate a solution f(x)=0.	
	# a,b : numbers	
	#     The interval in which to search for a solution. The function returns	
	#     None if f(a)*f(b) >= 0 since a solution is not guaranteed.	
	# N : (positive) integer	
	#     The number of iterations to implement.	
	#	
	# Returns	
	# -------	
	# x_N : number	
	#     The midpoint of the Nth interval computed by the bisection method. The	
	#     initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some	
	#     midpoint m_n = (a_n + b_n)/2, then the function returns this solution.	
	#     If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any	
	#     iteration, the bisection method fails and return None.	
	while 1:
		try:
			if f(a)*f(b) >= 0:	
				a = a - 10	
				b = b + 10	
				# print("Bisection method fails.")	
				continue	
			else:	
				break
		except Exception as e:
			return None
	a_n = a	
	b_n = b	
	for n in range(1,N+1):	
		m_n = (a_n + b_n)/2	
		f_m_n = f(m_n)	
		if f(a_n)*f_m_n < 0:	
			a_n = a_n	
			b_n = m_n	
		elif f(b_n)*f_m_n < 0:	
			a_n = m_n	
			b_n = b_n	
		elif np.abs(f_m_n) <= 1e-5:	
			# print("Found exact solution.")	
			return m_n	
		else:	
			# print("Bisection method fails.")	
			return None	
	return (a_n + b_n)/2	


def get_random_token_replacement(inputs, vocab, indices, replace_tokens, distinct):
	'''
	inputs is numpy array with indices (batch, max_len)
	grads is numpy array (batch, max_len, vocab_size)
	vocab is Vocab object
	indices is numpy array of size batch
	'''
	rand_replacements = {}    
	for i in range(inputs.shape[0]):
		inp = inputs[i]
		index = str(indices[i])
		
		d = {}      
		for repl_tok in replace_tokens:
			repl_tok_idx = vocab.stoi[repl_tok]
			if repl_tok_idx not in inp:
				continue
				
			exclude = list(d.values()) if distinct else []
			
			rand_idx = random.randint(0, len(vocab)-1)
			while not valid_replacement(vocab.itos[rand_idx], exclude=exclude):
				rand_idx = random.randint(0, len(vocab)-1)

			d[repl_tok] = vocab.itos[rand_idx]

		if len(d)>0:
			rand_replacements[index] = d
	
	return rand_replacements

def get_random_token_replacement_2(inputs, vocab, indices, replace_tokens, distinct, z_epsilon):

	rand_replacements = {}
	for i in range(inputs.shape[0]):
		inp = inputs[i]
		index = str(indices[i])
		d = {}

		# find all replace tokens in input i
		replace_tokens_i = []
		for repl_tok in replace_tokens:
			repl_tok_idx = vocab.stoi[repl_tok]
			if repl_tok_idx in inp:
				replace_tokens_i.append(repl_tok)

		if z_epsilon == 0:
			sites_picked = len(replace_tokens_i)
		else:
			sites_picked = min(len(replace_tokens_i), z_epsilon)

		random_sites = random.sample(replace_tokens_i, sites_picked)

		# replace sites with random tokens
		for site in random_sites:
			exclude = list(d.values()) if distinct else []
			rand_idx = random.randint(0, len(vocab)-1)
			while not valid_replacement(vocab.itos[rand_idx], exclude=exclude):
				rand_idx = random.randint(0, len(vocab)-1)
			d[site] = vocab.itos[rand_idx]

		rand_replacements[index] = d

	return rand_replacements
			

		



def get_exact_matches(data_loader, n_samples, model, id_to_label):

	"""
	Returns the filenames of samples whose predicted target sequence
	is equal to the actual target sequence.
	"""
	batch_iterator = iter(data_loader)
	model.eval()
	exact_matches = []
	special_tokens = ["<PAD>", "<SOS>", "<EOS>"]

	for bid, batch in enumerate(tqdm.tqdm(batch_iterator, total=n_samples)):

		filenames = batch.filenames
		# outputs_oho shape: [target len; batch size; vocab size]
		outputs_oho = model(batch.context, batch.contexts_per_label, batch.labels.shape[0], batch.labels)

		for i, fname in enumerate(filenames):
			output_i = outputs_oho[:,i,:]
			label = batch.labels[:,i].cpu().numpy().tolist()
			pred_tgt_seq = [id_to_label[np.argmax(output_i[j,:].detach().numpy())] for j in range(output_i.shape[0])]
			pred_tgt_seq = [tok for tok in pred_tgt_seq if tok not in special_tokens]
			ground_truth = [id_to_label[tok] for tok in label if id_to_label[tok] not in special_tokens]

			if pred_tgt_seq == ground_truth:
				exact_matches.append(index)

	return exact_matches
			


"""
def get_best_token_replacement(inputs, grads, vocab, indices, replace_tokens, distinct):
	'''
	inputs is numpy array with input vocab indices (batch, max_len)
	grads is numpy array (batch, max_len, vocab_size)
	vocab is Vocab object
	indices is numpy array of size batch
	returns a dict with {index: {"@R_1@":'abc', ...}}
	'''
	def valid_replacement(s, exclude=[]):
		return classify_tok(s)=='WORDS' and s not in exclude
	
	best_replacements = {}    
	for i in range(inputs.shape[0]):
		inp = inputs[i]
		gradients = grads[i]
		index = str(indices[i])
		
		d = {}				
		for repl_tok in replace_tokens:
			repl_tok_idx = input_vocab.stoi[repl_tok]
			if repl_tok_idx not in inp:
				continue
				
			inp[0] = repl_tok_idx
			mask = inp==repl_tok_idx

			# Is mean the right thing to do here? 
			avg_tok_grads = np.mean(gradients[mask], axis=0)

			exclude = list(d.values()) if distinct else []
			
			max_idx = np.argmax(avg_tok_grads)
			if not valid_replacement(vocab.itos[max_idx], exclude=exclude):
				idxs = np.argsort(avg_tok_grads)[::-1]
				for idx in idxs:
					if valid_replacement(vocab.itos[idx], exclude=exclude):
						max_idx = idx
						break
			d[repl_tok] = vocab.itos[max_idx]

		if len(d)>0:
			best_replacements[index] = d
	
	return best_replacements


def get_random_token_replacement(inputs, vocab, indices, replace_tokens, distinct):
	'''
	inputs is numpy array with input vocab indices (batch, max_len)
	grads is numpy array (batch, max_len, vocab_size)
	vocab is Vocab object
	indices is numpy array of size batch
	'''
	def valid_replacement(s, exclude=[]):
		return classify_tok(s)=='WORDS' and s not in exclude
	
	rand_replacements = {}    
	for i in range(inputs.shape[0]):
		inp = inputs[i]
		index = str(indices[i])
		
		d = {}		
		for repl_tok in replace_tokens:
			repl_tok_idx = input_vocab.stoi[repl_tok]
			if repl_tok_idx not in inp:
				continue
				
			inp[0] = repl_tok_idx

			exclude = list(d.values()) if distinct else []
			
			rand_idx = random.randint(0, len(vocab)-1)
			while not valid_replacement(vocab.itos[rand_idx], exclude=exclude):
				rand_idx = random.randint(0, len(vocab)-1)

			d[repl_tok] = vocab.itos[rand_idx]

		if len(d)>0:
			rand_replacements[index] = d
	
	return rand_replacements
"""
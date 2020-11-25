import json
import gzip
from scipy import stats
import os
import re

def count_tokens(filename):

	with open(filename, 'r') as f:
		f_lines = f.readlines()

	src_idx = f_lines[0].strip().split('\t').index('src')
	fname_idx = f_lines[0].strip().split('\t').index('from_file')
	tokens = {}
	for l in f_lines[1:]:
		l = l.strip().split('\t')
		l_tokens = l[src_idx].split(' ')
		n = len(l_tokens)
		tokens[l[fname_idx]] = n
	return tokens

def count_sites(fname):
	with open(fname, 'r') as f:
		flines = f.readlines()
	src_idx = flines[0].strip().split('\t').index('src')

	count = []
	for l in flines[1:]:
		l = l.strip().split('\t')
		l = l[src_idx]
		count.append(len(set(re.compile('replaceme\d+').findall(l))))
	return count, stats.describe(count)

def get_site_stats(orig_site_map, optimal_replacements, idx_to_fname, exact_matches, tokens_count):

	transforms = ['InsertPrintStatements', 'AddDeadCode', 'RenameLocalVariables', 'RenameParameters', 'RenameFields', 'ReplaceTrueFalse']
	total_sites_count = []
	transforms_picked = {'transforms.'+t:[] for t in transforms}
	optim_sites_count = []

	for f_idx in exact_matches:

		f_idx = str(f_idx)
		f_name = idx_to_fname[f_idx]
		# if tokens_count[f_name] <= 250:
		f_sites = orig_site_map[f_name]
		total_sites_count.append(len(f_sites))

		for t in transforms_picked:
			transforms_picked[t].append(0)
		optim_sites_count.append(0)

		if f_idx in optimal_replacements:
			for site1 in optimal_replacements[f_idx]:
				for site2 in f_sites:
					if site1 in site2:

						if site1 != site2 or optimal_replacements[f_idx][site1] != f_sites[site2][0]:
							transforms_picked[f_sites[site2][1]][-1] += 1
							optim_sites_count[-1] += 1

						break

	# assert 0 not in optim_sites_count

	total_sites_stats = stats.describe(total_sites_count)
	transforms_stats = {t:stats.describe(transforms_picked[t]) for t in transforms_picked}
	optim_sites_stats = stats.describe(optim_sites_count)
	return total_sites_stats, transforms_stats, optim_sites_stats




def make_hist(x, bins, xlabel, fname):

	plt.hist(x, bins)
	plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(len(x)))
	plt.xlabel(xlabel)
	plt.savefig(fname)
	plt.clf()



if __name__ == '__main__':

	for dataset in ['c2s/java-small']:
		for transform_name in ['transforms.Combined']:
			src_dir = './datasets/transformed/preprocessed/tokens/{}/{}/'.format(dataset, transform_name)
			site_map_path = src_dir + 'test_site_map.json'
			site_map = json.load(open(site_map_path, 'r'))
			file_name = 'test.tsv'
			# get distribution of sites in the whole test set (20K samples)
			# print(dataset, transform_name)
			site_count, site_stats = count_sites(src_dir+file_name)
			# print(len(site_count), site_stats)

			tokens = count_tokens(src_dir+file_name)

			experiments = ['v2-6-z_o_5-pgd_3_no-transforms.Combined-javasmall-check', 'v2-4-z_o_1-pgd_3_no-transforms.Combined-javasmall-check']
			for experiment_name in experiments:
				# experiment_name = 'v2-6-z_o_5-pgd_3_no-transforms.Combined-javasmall-check'
				adv_dir = './datasets/adversarial/{}/tokens/{}/'.format(experiment_name, dataset)
				idx_to_fname = json.load(open(adv_dir+'test_idx_to_fname.json', 'r'))
				optimal_replacements = json.load(open(adv_dir + 'targets-test-gradient.json', 'r'))
				optimal_replacements = optimal_replacements[transform_name]
				exact_matches = json.load(open(adv_dir+'exact_matches_idxs.json', 'r'))
				# get stats on exact matches (~3K)
				total_sites_stats, transform_stats, optim_sites_stats = get_site_stats(site_map, optimal_replacements, idx_to_fname, exact_matches, tokens)
				print(total_sites_stats, '\n', transform_stats, '\n', optim_sites_stats)







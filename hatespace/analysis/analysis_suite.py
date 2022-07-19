import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import lsq_linear, linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.special import psi, polygamma, betainc, gamma
import ast
import matplotlib.pyplot as plt
from scipy import special
import time
import logging
import seaborn as sns
import threading
from typing import Optional, Tuple, Union
from hatespace.datasets import IronMarch
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from hatespace.analysis.dirichlet_tools import DirichletGOF
import os
from datetime import datetime
import warnings
import itertools


class IronmarchAnalysis:

	def __init__(self,
		dataset_path: str,
		dataset = None, 
		latent_vectors_file_path: Optional[str] = '', 
		latent_vectors: Optional[np.ndarray] = None,
		values_dict: Optional[dict] = {}
	):

		self.dataset_path = dataset_path

		if dataset is None:
			self.dataset = IronMarch(dataset_path)
		else:
			self.dataset = dataset


		if values_dict != {}:
			self.values_dict = values_dict
			self.latent_vectors_split = []
			self.unix_timestamps = []
			self.ymd_timestamps = []
			self.post_ids = []
			self.posts = []
			self.authors = []

			for x in values_dict['data']:
				self.latent_vectors_split.append(x['latent_vectors'])
				self.post_ids.append(x['post_ids'])
				self.unix_timestamps.append(x['unix_timestamps'])
				self.ymd_timestamps.append(x['ymd_timestamps'])
				self.posts.append(x['posts'])
				self.authors.append(x['authors'])

			if len(self.latent_vectors_split) == 0 or self.latent_vectors_split[0].shape[0] == 0:
				raise ValueError('No posts with specified conditions. Specified time range is outside the original time range and/or given author(s) have no posts in time range.')

			self.latent_vectors = np.concatenate(self.latent_vectors_split, axis=0)
			self.unix_timestamps = list(itertools.chain(*self.unix_timestamps))
			self.ymd_timestamps = list(itertools.chain(*self.ymd_timestamps))
			self.post_ids = list(itertools.chain(*self.post_ids))
			self.posts = list(itertools.chain(*self.posts))
			self.authors = list(itertools.chain(*self.authors))

			self.forums = values_dict['forums']
			self.msgs = values_dict['msgs']

			self.id_index_dict = dict((value, idx) for idx,value in enumerate(self.post_ids))

		else:

			if latent_vectors is not None:
				self.latent_vectors = latent_vectors
			elif latent_vectors_file_path is not '':
				try:
					self.latent_vectors = np.load(latent_vectors_file_path)
				except:
					print('Latent vectors file must by .npy file')
			else:
				raise ValueError('Must provide path to a latent vectors file, or provide a latent vectors array directly')

			self.posts = [x['data'] for x in self.dataset]
			self.posts = [str(i) for i in self.posts]

			self.post_ids = [x['id'] for x in self.dataset] 
			self.id_index_dict = dict((value, idx) for idx,value in enumerate(self.post_ids))

			self.forums, self.msgs = self.return_df_from_ids(
				forums = pd.read_csv(os.path.join(dataset_path, 'core_search_index.csv')),
				msgs = pd.read_csv(os.path.join(dataset_path, 'core_message_posts.csv')),
				post_ids = self.post_ids
			)

			self.latent_vectors, self.unix_timestamps, self.ymd_timestamps, self.post_ids, self.posts, self.authors = self.return_sorted(self.forums, self.msgs)

			self.latent_vectors_split = [self.latent_vectors]
			
			self.values_dict = {'data': [self.make_data_dict(latent_vectors, unix_timestamps, ymd_timestamps, post_ids, posts, authors)], 'forums': self.forums, 'msgs': self.msgs}
			
		self.latent_dim_size = self.latent_vectors.shape[1]


	def index_by_indices(self, list_to_index, indices):
		return [list_to_index[i] for i in indices]

	def get_posts_from_post_ids(self, post_ids = []):
		posts = [self.posts[self.id_index_dict[x]] for x in post_ids]
		return posts

	def return_sorted(self, forums, msgs):
		forum_ids = ['forum_post-' + str(x) for x in forums['index_id'].tolist()]
		msg_ids = ['direct_message-' + str(x) for x in msgs['msg_id'].tolist()]

		post_ids = forum_ids + msg_ids
		timestamps = forums['index_date_created'].tolist() + msgs['msg_date'].tolist()
		authors = forums['index_author'].tolist() + msgs['msg_author_id'].tolist()

		indices = [self.id_index_dict[x] for x in post_ids]

		latent_vectors = self.latent_vectors[indices]

		sort_indices = np.argsort(timestamps)
		latent_vectors = latent_vectors[sort_indices]
		unix_timestamps = self.index_by_indices(timestamps, sort_indices)
		ymd_timestamps = [datetime.fromtimestamp(x).strftime('%Y/%m/%d') for x in timestamps]
		post_ids = self.index_by_indices(post_ids, sort_indices)
		posts = self.get_posts_from_post_ids(post_ids)
		authors = self.index_by_indices(authors, sort_indices)

		return latent_vectors, unix_timestamps, ymd_timestamps, post_ids, posts, authors

	def return_df_from_ids(self, forums, msgs, post_ids):
		forum_ids = [int(x.replace('forum_post-', '')) for x in post_ids if 'forum_post' in x]
		msg_ids = [int(x.replace('direct_message-', '')) for x in post_ids if 'direct_message' in x]

		forums = forums[forums['index_id'].isin(forum_ids)].sort_values(by='index_date_created')
		msgs = msgs[msgs['msg_id'].isin(msg_ids)].sort_values(by='msg_date')

		return forums, msgs

	def get(self,
		start_time: Optional[Union[Tuple[int,int,int], int]] = None, 
		end_time: Optional[Union[Tuple[int,int,int], int]] = None, 
		author_ids: Optional[list] = [],
		split_by = ''
		):
		

		if type(start_time) == tuple:
			start = int(datetime(start_time[0], start_time[1], start_time[2]).timestamp())
		elif type(start_time) == int:
			start = start_time
		elif start_time == None:
			start = -1
		else:
			warnings.warn('Provide either a tuple (year, month, day) or a UNIX timestamp. start_time is defaulted to beginning of dataset.')
			start = -1

		if type(end_time) == tuple:
			end = int(datetime(end_time[0], end_time[1], end_time[2]).timestamp())
		elif type(end_time) == int:
			end = end_time
		elif end_time == None:
			end = 1e99
		else:
			warnings.warn('Provide either a tuple (year, month, day) or a UNIX timestamp. end_time is defaulted to end of dataset.')

		forums = self.forums[(self.forums['index_date_created'] > start) & (self.forums['index_date_created'] < end)]
		msgs = self.msgs[(self.msgs['msg_date'] > start) & (self.msgs['msg_date'] < end)]

		if len(author_ids) > 0:
			forums = forums[forums['index_author'].isin(author_ids)]
			msgs = msgs[msgs['msg_author_id'].isin(author_ids)]

		latent_vectors, unix_timestamps, ymd_timestamps, post_ids, posts, authors = self.return_sorted(forums, msgs)

		if split_by != '':
			if split_by.lower() == 'day':
				dates = [datetime.fromtimestamp(x).strftime('%Y/%m/%d') for x in timestamps]
			elif split_by.lower() == 'month':
				dates = [datetime.fromtimestamp(x).strftime('%Y/%m') for x in timestamps]
			current = None
			split_indices = []

			for idx, date in enumerate(dates):
				if current != date:
					current = date
					time_indices = []
					split_indices.append(time_indices)
				time_indices.append(idx)

			split_dict = []
			for split in split_indices:
				split_latent_vectors = latent_vectors[split]
				split_unix_timestamps = self.index_by_indices(unix_timestamps, split)
				split_ymd_timestamps = self.index_by_indices(ymd_timestamps, split)
				split_ids = self.index_by_indices(post_ids, split)
				split_posts = self.get_posts_from_post_ids(split_ids)
				split_authors = self.index_by_indices(authors, split)

				split_forums, split_msgs = self.return_df_from_ids(forums, msgs, split_ids)

				split_dict.append(self.make_data_dict(split_latent_vectors, split_unix_timestamps, split_ymd_timestamps, split_ids, split_posts, split_authors))

			vals_dict = {'data': split_dict, 'forums': forums, 'msgs': msgs}
			
		else:

			vals_dict = {'data': [self.make_data_dict(latent_vectors, unix_timestamps, ymd_timestamps, post_ids, posts, authors)], 'forums': forums, 'msgs': msgs}

		return IronmarchAnalysis(dataset_path = self.dataset_path,
			dataset = self.dataset,
			values_dict=vals_dict)

	def make_data_dict(self, latent_vectors, unix_timestamps, ymd_timestamps, post_ids, posts, authors):
		return {'latent_vectors': latent_vectors, 'unix_timestamps': unix_timestamps, 'ymd_timestamps': ymd_timestamps, 'post_ids': post_ids, 'posts': posts, 'authors': authors}


	def get_nearest_indices(self, num_vectors_per_at: int) -> np.ndarray:
		'''
		Returns array of indices closest to all archetypes.

		'''

		# Computes distances from latent vectors to every corner of simplex
		nearest_indices = []
		for latent in self.latent_vectors_split:
			dists = np.zeros((self.latent_dim_size, latent.shape[0]))
			for i, vertex in enumerate(np.eye(self.latent_dim_size)):
				dists[i] = np.sqrt(np.sum(np.square(latent-vertex), axis=1))

			# Get closest indices to each archetype by sorting
			nearest_indices.append(np.argsort(dists)[:, :num_vectors_per_at])

		return nearest_indices


	# Gets specified number of archetypal posts
	# Only returns list of lists containing the posts. Update to have pandas option
	def get_archetypal_posts(self, num_posts_per_at: int, save_to_folder = ''):
		all_at_posts = []
		all_at_timestamps = []
		all_at_authors = []

		nearest_indices = self.get_nearest_indices(num_posts_per_at)

		dir = os.path.join(save_to_folder, 'archetypal_posts')
		if os.path.exists(dir):
		    shutil.rmtree(dir)
		os.makedirs(dir)
		
		for idx, (indices, latent) in enumerate(zip(nearest_indices, self.latent_vectors_split)):
			with open(os.join(dir, 'top_archetypal_{}.txt'.format(idx)), 'w') as f:
				at_posts = []
				at_timestamps = []
				at_authors = []
				for i in range(self.latent_dim_size):
					f.write('===========Archetype {}==========='.format(i))
					top_posts = []
					top_timestamps = []
					top_authors = []

					for k in indices[i]:
						f.write('{} -- {} -- Author {} -- {}'.format(k, self.ymd_timestamps[k], self.authors[k], self.posts[k]))
						top_posts.append(self.posts[k])
						top_timestamps.append(self.ymd_timestamps[k])
						top_authors.append(self.authors[k])

					at_posts.append(top_posts)
					at_timestamps.append(top_timestamps)
					at_authors.append(top_authors)

				all_at_posts.append(at_posts)
				all_at_posts.append(at_timestamps)
				all_at_posts.append(at_authors)

		return all_at_posts, at_timestamps, at_authors

	# Keyword extraction using TF-IDF algorithm. Return as pandas array
	# TODO: Provide a visualization that sorts TF-IDF scores for each archetype and makes a bar plot
	def archetypes_tfidf_scores(self, num_posts_per_at: int):
		# For each archetype, concatenate all posts into one string.
		# We will take all the posts in a single archetype to be one document
		nearest_indices = self.get_nearest_indices(num_posts_per_at)
		at_posts = []
		for i in range(self.latent_dim_size):
			top_posts = [self.posts[k] for k in nearest_indices[i]]
			post = ''
			for j in range(20):
				top = top_posts[j].lower()
				top = re.sub(r"<url>", ' ', top) # Can probably combine these two re lines
				top = re.sub('[\n\t\r]', ' ', top)
				post += top
				post += ''
			at_posts.append(post)

		my_stop_words = text.ENGLISH_STOP_WORDS
		vectorizer = TfidfVectorizer(stop_words=my_stop_words)
		vectors = vectorizer.fit_transform(at_posts)

		feature_names = vectorizer.get_feature_names()
		dense = vectors.todense()
		denselist = dense.tolist()
		df = pd.DataFrame(denselist, columns=feature_names)

		return df


	# TODO: Some way to obtain weights and biases of side information linear head
	def get_weights_and_biases():
		pass
		

	# Visualizes pairwise angles between hyperplane normal vectors obtained from linear head
	def normal_hyperplanes(self, weights):
		class_names = ['nationality', 'ethnicity', 'religion', 'gender', 'sexual_orientation', 'disability', 'class']
		
		weights_norm = weights / np.linalg.norm(weights, axis=1)[:, np.newaxis]

		cos_similarity = np.zeros((7,7))
		for i in range(7):
		  for j in range(7):
		    cos_similarity[i][j] = np.dot(weights_norm[i], weights_norm[j])

		sns.heatmap(cos_similarity, cmap=sns.cm.rocket_r, xticklabels=class_names, yticklabels=class_names)

		closet_ats = []
		for normal in weights_norm:
			closet_ats.append(np.argmax(np.dot(np.eye(self.latent_dim_size, normal))))

		return cos_similarity, closet_ats


	# Get proportion of posts that lie on either side of hyperplane generated by side info
	def compute_hyperplane_proportions(self, weights, biases):
		proportions = []
		for i in range(7):
			count = 0
			for j in range(self.latent_vectors.shape[0]):
				if np.dot(weights[i], self.latent_vectors[j]) + biases[i] > 0:
					count += 1
			proportions.append(count / self.latent_vectors.shape[0])

		print('Proportion of posts above each hyperplane')
		for idx, i in enumerate(class_names):
			print(class_names[idx] + ': ' + str(proportions[idx]))


	# TODO: implement full dirichlet analysis of latent space
	def dirichlet_analysis(self, significance_level, dim, n_iter=10, sample_size=100, crit_val_arr=None, print_log=False):
		gof = DirichletGOF(significance_level, dim, sample_size, crit_val_arr)

		results = gof.test_statistic(self.latent_vectors, n_iter=n_iter, print_log=print_log)

		return results

	def __getitem__(self, key):
		return self.values_dict[key]

	def __str__(self):
		printout = 'Dictionary object with keys: "data", "forums", "msgs"\nValue of "data" is a list containing {} element(s).\nEach element of the list is a dictionary with keys: "latent_vectors", "post_ids", "timestamps", "posts"'.format(len(self.values_dict['data']))
		return printout

	def __repr__(self):
		printout = 'Dictionary object with keys: "data", "forums", "msgs"\nValue of "data" is a list containing {} element(s).\nEach element of the list is a dictionary with keys: "latent_vectors", "post_ids", "timestamps", "posts"'.format(len(self.values_dict['data']))
		return printout


	'''
	track dirichlet-ness over time
	track centroid movement over time (all users/certain bad actors)
	use kernel density estimator to make plot along each archetypal direction (seaborn)
	csv of embedded coordinates, time, user id, location - maybe average each coordinate according to location
	'''

	'''
	[DONE] Get from author ids
	[DONE] Nearest posts per time period
	[DONE] cosine simililarity heat map
	[DONE] archetypes pointing towards
	have "get" return IronmarchAnalysis instance.
	rectangle - xaxis = time, yaxis = proportion of archetype
		each post has proportions of archetype
	'''
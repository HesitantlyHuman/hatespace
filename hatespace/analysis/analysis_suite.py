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


class IronmarchAnalysis:

	def __init__(self,
		dataset_path: str, 
		latent_vector_file_path: Optional[str] = '', 
		latent_vectors: Optional[np.ndarray] = None
	):

		if latent_vectors is not None:
			self.latent_vectors = latent_vectors
		elif path_to_latent_vector_file is not '':
			try:
				self.latent_vectors = np.load(latent_vector_file_path)
			except:
				print('Latent vectors file must by .npy file')
		else:
			raise ValueError('Must provide path to a latent vectors file, or provide a latent vectors array directly')

		dataset = IronMarch(dataset_path)
		self.msgs = pd.read_csv(os.path.join(dataset_path, 'core_message_posts.csv'))
		self.forums = pd.read_csv(os.path.join(dataset_path, 'core_search_index.csv'))

		self.posts = [x['data'] for x in dataset]
		self.posts = [str(i) for i in self.posts]
		
		self.post_ids = [x['id'] for x in dataset] 

		self.id_index_dict = dict((value, idx) for idx,value in enumerate(self.post_ids))


	def get_posts_from_post_ids(self, post_ids = []):
		posts = [self.posts[self.id_index_dict[x]] for x in post_ids]
		return posts


	def get(self, 
		unix_start_time: Optional[int] = None, 
		unix_end_time: Optional[int] = None,
		start_ymd: Optional[tuple] = None, 
		end_ymd: Optional[tuple] = None, 
		author_ids: Optional[list] = [],
		split_by = ''
		):
		

		if unix_start_time is not None:
			start = unix_start_time
		elif start_ymd is not None:
			start = int(datetime(start_ymd[0], start_ymd[1], start_ymd[2]).timestamp())
		else:
			start = -1

		if unix_end_time is not None:
			end = unix_end_time
		elif end_ymd is not None:
			end = int(datetime(end_ymd[0], end_ymd[1], end_ymd[2]).timestamp())
		else:
			end = 1e99


		ranged_forums = self.forums[(self.forums['index_date_created'] > start) & (self.forums['index_date_created'] < end)]
		ranged_msgs = self.msgs[(self.msgs['msg_date'] > start) & (self.msgs['msg_date'] < end)]

		if len(author_ids) > 0:
			ranged_forums = ranged_forums[ranged_forums['index_author'].isin(author_ids)]
			ranged_msgs = ranged_msgs[ranged_msgs['msg_author_id'].isin(author_ids)]

		forums_ids = ['forum_post-' + str(x) for x in ranged_forums['index_id'].tolist()]
		msgs_ids = ['direct_message-' + str(x) for x in ranged_msgs['msg_id'].tolist()]

		ranged_ids = forums_ids + msgs_ids
		timestamps = ranged_forums['index_date_created'].tolist() + ranged_msgs['msg_date'].tolist()

		indices = [self.id_index_dict[x] for x in ranged_ids]

		ranged_latent_vectors = self.latent_vectors[indices]

		sort_indices = np.argsort(timestamps)
		ranged_latent_vectors = ranged_latent_vectors[sort_indices]
		timestamps = [timestamps[i] for i in sort_indices]
		ranged_ids = [ranged_ids[i] for i in sort_indices]

		posts = self.get_posts_from_post_ids(ranged_ids)

		if split_by != '':
			if split_by.lower() == 'day':
				dates = [datetime.fromtimestamp(x).strftime('%d/%m/%Y') for x in timestamps]
			elif split_by.lower() == 'month':
				dates = [datetime.fromtimestamp(x).strftime('%m/%Y') for x in timestamps]
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
				split_timestamps = [timestamps[i] for i in split]
				split_ids = [ranged_ids[i] for i in split]
				split_posts = self.get_posts_from_post_ids(split_ids)

				split_dict.append({'latent_vectors': split_latent_vectors, 'ids': split_ids, 'timestamps': split_timestamps, 'posts': split_posts})

			return split_dict

		return {'latent_vectors': ranged_latent_vectors,'sorted_ids': ranged_ids, 'sorted_timestamps': timestamps, 'sorted_posts': posts}


	def get_nearest_indices(self, num_vectors_per_at: int) -> np.ndarray:
		'''
		Returns array of indices closest to all archetypes.

		'''

		# Computes distances from latent vectors to every corner of simplex
		latent_dim_size = self.latent_vectors.shape[1]
		dists = np.zeros((latent_dim_size, self.latent_vectors.shape[0]))
		for i, vertex in enumerate(np.eye(latent_dim_size)):
			dists[i] = np.sqrt(np.sum(np.square(self.latent_vectors-vertex), axis=1))

		# Get closest indices to each archetype by sorting
		nearest_indices = np.argsort(dists)[:, :num_vectors_per_at]

		return nearest_indices


	# Gets specified number of archetypal posts
	# Only returns list of lists containing the posts. Update to have pandas option
	def get_archetypal_posts(self, num_posts_per_at: int):
		nearest_indices = self.get_nearest_indices(self.latent_vectors, num_posts_per_at)
		latent_dim_size = self.latent_vectors.shape[1]
		at_posts = []
		for i in range(latent_dim_size):
		    top_posts = [texts[k] for k in nearest_indices[i]]
		    at_posts.append(top_posts)
		return at_posts


	# Keyword extraction using TF-IDF algorithm. Return as pandas array
	# TODO: Provide a visualization that sorts TF-IDF scores for each archetype and makes a bar plot
	def archetypes_tfidf_scores(self, num_posts_per_at: int):
		# For each archetype, concatenate all posts into one string.
		# We will take all the posts in a single archetype to be one document
		nearest_indices = self.get_nearest_indices(self.latent_vectors, num_posts_per_at)
		latent_dim_size = self.latent_vectors.shape[1]
		at_posts = []
		for i in range(latent_dim_size):
			top_posts = [texts[k] for k in nearest_indices[i]]
			post = ''
			for j in range(20):
				top = top_posts[j].lower(0)
				top = re.sub(r"<url>", ' ', top) # Can probably combine these two re lines
				top = re.sub('[\n\t\r]', ' ', top)
				post += top
				post += ''
			at_posts.append(post)

		my_stop_words = text.ENGLISH_STOP_WORDS
		vectorizer = TfidfVectorizer(stop_words=my_stop_words)
		vectors = vectorizer.fit_transform(combined_at_posts)

		feature_names = vectorizer.get_feature_names()
		dense = vectors.todense()
		denselist = dense.tolist()
		df = pd.DataFrame(denselist, columns=feature_names)

		return df


	# TODO: Some way to obtain weights and biases of side information linear head
	def get_weights_and_biases():
		pass
		

	# Visualizes pairwise angles between hyperplane normal vectors obtained from linear head
	def compute_normal_hyperplanes(self, weights):
		class_names = ['nationality', 'ethnicity', 'religion', 'gender', 'sexual_orientation', 'disability', 'class']
		
		weights_norm = weights / np.linalg.norm(weights, axis=1)[:, np.newaxis]

		dot_prods = np.zeros((7,7))
		for i in range(7):
		  for j in range(7):
		    dot_prods[i][j] = np.dot(weights_norm[i], weights_norm[j])

		sns.heatmap(dot_prods, cmap=sns.cm.rocket_r, xticklabels=class_names, yticklabels=class_names)


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
	def dirichlet_analysis(self, significance_level, dim, sample_size, crit_val_arr=None):
		gof = DirichletGOF(significance_level, dim, sample_size, crit_val_arr=None)

		results = gof.test_statistic(self.latent_vectors)

		return results


	'''
	track dirichlet-ness over time
	track centroid movement over time (all users/certain bad actors)
	use kernel density estimator to make plot along each archetypal direction (seaborn)
	csv of embedded coordinates, time, user id, location - maybe average each coordinate according to location
	'''
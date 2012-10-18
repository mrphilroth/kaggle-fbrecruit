#!/usr/bin/env python

#from __future__ import division #seems to perform better w/integer division
import utilities
import operator
import collections
import multiprocessing

def make_recs(target_node): 
  sorted_recs = rank(target_node)
  return target_node, sorted_recs[:10]

def rank(target_node):
  scores = collections.defaultdict(int)
  
  forward_count = len(graph[target_node])
  back_count = len(graph_inverse[target_node])
  base_score = 1000000 / (forward_count + back_count)
  iterations = 2      # I tried using more iterations, but it did not improve my score
  for friend in graph[target_node]:
    scores[friend] += base_score * 0.8
    propagate_score(scores, friend, base_score, iterations)
  for back_friend in graph_inverse[target_node]:
    scores[back_friend] += base_score * 0.8
    propagate_score(scores, back_friend, base_score, iterations)
  return [node for node, score in sorted(scores.items(), key=operator.itemgetter(1), reverse=True) if node not in graph[target_node] and node != target_node]

def propagate_score(scores, node, score, iteration):
  if iteration == 0 or score == 0:
    return
  iteration -= 1
  
  forward_count = len(graph[node])
  back_count = len(graph_inverse[node])
  per_link = score / (forward_count + back_count)
  for dest in graph[node]:
    scores[dest] += per_link * 0.8   # Changing this parameter to 0.8 improved my score
    propagate_score(scores, dest, per_link, iteration)
  for back_dest in graph_inverse[node]:
    propagate_score(scores, back_dest, per_link, iteration)

def run_recs(train_file, test_file, submission_file):
    global graph, graph_inverse
    graph, graph_inverse = utilities.read_graph_and_inverse(train_file)
    test_nodes = utilities.read_nodes_list(test_file)
    
    #change the val below to match cpu/memory usage, allow for ~1.2G of ram per cpu, swap will kill performance
    pool = multiprocessing.Pool(8)
    predictions = {}
    for target_node, recs in pool.imap_unordered(make_recs, test_nodes, chunksize=10000): #can experiment w/chunksize
      predictions[target_node] = recs
    test_predictions = [predictions[node] for node in test_nodes]
    utilities.write_submission_file(submission_file, 
                                    test_nodes, 
                                    test_predictions)

if __name__ == "__main__":
    run_recs("../data/train.csv",
             "../data/test.csv",
             "../submissions/submission_edgerank_iter2_80percent.csv")
    

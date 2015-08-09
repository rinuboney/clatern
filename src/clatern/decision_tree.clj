(ns clatern.decision-tree
  (:refer-clojure :exclude [partition])
  (:require [clojure.set :refer [union]]
            [clatern.protocols :refer [ClassProbabilityEstimator]]
            [clojure.core.matrix :refer :all]))

(defn- map-values
  [f m]
  (into {} (for [[k v] m] [k (f v)])))

(defn- gini-impurity
  "The Gini index or impurity is the probability of the elements
  being mislabeled if the labels were chosen randomly from the label
  distribution of the set.

  `y` is a vector of element labels, and `elems` is a vector of
  indices of the elements on which to calculate the Gini index."
  [y elems]
  (let [n (count elems)
        label-counts (frequencies (map y elems))
        label-freq (map-values #(/ % n) label-counts)
        labels (set (keys label-counts))
        label-probs (map #(* (label-freq %)
                             (label-freq %)) labels)
        gini (- 1.0 (reduce + label-probs))]
    gini))

(defn- partition
  "Partition samples with indices given by `elems` into left and right.
  Left contains the elements whose values for feature `feature-idx` are
  <= `threshold` and right contains the rest.

  `X` is a feature matrix of `n_samples` and `n_features`. `y` is a vector
  of sample labels.  `elems` is a vector of sample indices to consider.
  `feature-idx` is the index for the feature used in the partitioning.
  `threshold` is the value to split the partitions by."
  [X y elems feature-idx threshold]
  (let [feature-values (map (get-column X feature-idx) elems)
        values-elem (map vector feature-values elems)
        groups (group-by #(if (<= (first %) threshold) :left :right) values-elem)
        group-elem (map-values #(map second %) groups)]
    [(group-elem :left) (group-elem :right)]))

(defn- find-optimal-threshold
  "Find the threshold which minimizes the cost of the split. 

  `X` is a feature matrix of `n_samples` and `n_features`, `y` is a
  vector of `n_samples` labels, `elems` are the sample indices to
  use in the cost calculation, and `feature-idx` is the feature to
  threshold."
  [X y elems feature-idx]
  (let [n_samples (count elems)
        feature-values (map (get-column X feature-idx) elems)
        values-elem (map vector feature-values elems)
        thresholds (sort (set feature-values))
        part-fn #(partition X y elems feature-idx %)
        threshold-partitions (map #(vector % (part-fn %)) thresholds)
        threshold-costs (for [[t [left right]] threshold-partitions]
                          [t 
                           (+ (gini-impurity y left) (gini-impurity y right))
                           left
                           right])]
    (apply min-key second threshold-costs)))
       
(defn- find-best-split
  "Iterates over all features and thresholds for each feature to find the 
  split which minimizes the cost.

  `X` is a `n_samples` x `n_features` feature matrix. `y` is a vector of
  labels.  `feature-sampler` is a function that takes the number of features
  and returns a sequence of feature indices. `elems` is a vector of sample
  indices to use."
  [X y sample-features elems]
  (let [n-features (dimension-count X 1)
        features (sample-features n-features)
        feature-optimal-thresholds (for [idx features]
                                     [idx (find-optimal-threshold X y elems idx)])
        [feature-idx [threshold cost left right]] (apply min-key #(second (second %))
                                                         feature-optimal-thresholds)]
    [feature-idx threshold cost left right]))

(defn- split-node?
  "Determine whether or not to split the node or consider the node as a leaf.

  `y` is a vector of class labels.  `max-depth` is the max depth of the tree.
  `depth` is the depth of the current node. `left` are the indices of the left
  child and `right` are the indices of the right child."
  [y max-depth depth left right]
  (and (<= depth max-depth)
       (< 0 (count left))
       (< 0 (count right))
       (< 1 (count (union (set (map y left))
                          (set (map y right)))))))
        
(defn- train-tree
  "Train a decision tree of `max-depth` from the feature-matrix `X` and the
  labels `y`.

  `X` is a `n_samples` x `n_features` feature matrix. `y` is a vector of labels.
  `max-depth` is the maximum depth of the tree. `feature-sampler` is a function
  that takes the number of features and returns a sequence of feature indices.
  `elems` is a list of sample indices to train the tree on -- indices can appear
  multiple times."
  ([X y max-depth feature-sampler elems]
   (train-tree X y max-depth 1 feature-sampler elems))
  ([X y max-depth depth feature-sampler elems]
   (let [n-elems (count elems)
         [feature-idx threshold cost left right] (find-best-split X y feature-sampler elems)
         default-counts (into {} (map #(vector % 0) y))
         label-counts (merge default-counts (frequencies (map y elems)))]
     (if (split-node? y max-depth (inc depth) left right)
       {:type :internal,
        :threshold threshold,
        :feature-idx feature-idx,
        :left (train-tree X y max-depth (inc depth) feature-sampler left),
        :right (train-tree X y max-depth (inc depth) feature-sampler right)}
       {:label-counts label-counts,
        :type :leaf}))))

(defn- find-leaf-node
  "Find leaf node for vector `v` in the decision tree `tree`."
  [tree v]
  (if (= (tree :type) :leaf)
    tree
    (let [{feature-idx :feature-idx,
           threshold :threshold} tree
           feature-value (v feature-idx)
           child (if (<= feature-value threshold) :left :right)]
      (find-leaf-node (tree child) v))))

(defn- normalize
  "Normalize counts"
  [label-counts]
  (let [leaf-elems (reduce + (vals label-counts))]
    (map-values #(/ % leaf-elems) label-counts)))

(defn- predict
  "Predicts the class of vector `v` using a majority vote of the class distribution
  from the leaf of a decision tree `tree`."
  [tree v]
  (let [leaf (find-leaf-node tree v)]
    (first (apply max-key second (leaf :label-counts)))))

(defn- predict-prob
  "Predicts the class label probability distribution for the vector `v`."
  [tree v]
  (let [leaf (find-leaf-node tree v)]
        (normalize (leaf :label-counts))))

(defn- predict-log-prob
  "Predicts the class label log probabilities for the vector `v`."
  [tree v]
  ; prevent log(0)
  (map-values #(java.lang.Math/log (+ % 0.000001)) (predict-prob tree v)))

(defrecord DecisionTree [tree]
  clojure.lang.IFn
  (invoke [this v] (predict tree v))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args))

  ClassProbabilityEstimator
  (predict-prob [this v] (predict-prob tree v))
  (predict-log-prob [this v] (predict-log-prob tree v)))
      
(defn decision-tree
  "Trains decision tree and returns model.

  Implementation of the CART algorithm that uses Gini impurity to determine
  the best splits and majority vote on the leaves' class label distributions
  for predictions.  

  `X` is a `n_samples` x `n_features` feature matrix. `y` is a vector of a
  class labels. `max-depth` is optional (default 5) and controls the
  maximum height of the tree. `elems` is an (optional) list of samples to
  use to train the tree -- indices can occur multiple times."
  [X y & {:keys [max-depth elems]
          :or {max-depth 5 elems (range (count y))}}]
  (let [tree (train-tree X y max-depth range elems)]
    (DecisionTree. tree)))

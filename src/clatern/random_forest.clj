(ns clatern.random-forest
  (:require [clatern.decision-tree :refer [cart]]
            [clatern.protocols :as p]
            [clatern.utils :refer [map-values]]))

(defn- sample-with-replacement
  "Sample `n-samples` values with replacement from the integers
  0 (inclusive) to `n-indices` (exclusive)."
  [n-samples n-indices]
  (repeatedly n-samples (partial rand-int n-indices)))

(defn- sample-without-replacement
  "Sample `n-samples` values without replacement from the integers
  0 (inclusive) to `n-indices` (exclusive)."
  [n-samples n-indices]
  (take n-samples (shuffle (range n-indices))))

(defn- sqrt-round
  "Return at least 1 or the rounded square root of `n-features`."
  [n-features]
  (max 1 (Math/round (Math/sqrt n-features))))

(defn- train-forest
  "Train a random forest of `n-trees` with `max-depth` from the
  feature matrix `X` and class labels `y`"
  [X y n-trees max-depth]
  (let [n-elems (count y)]
    (for [i (range n-trees)]
      (cart X y
            :max-depth max-depth 
            :feature-sampler #(sample-without-replacement (sqrt-round %) %)
            :elems (sample-with-replacement n-elems n-elems)))))

(defn- average-values
  [maps]
  (map-values #(/ % (count maps)) (apply merge-with + maps)))

(defn- predict-prob
  "Predict the probabilities of `v` belonging to each class using
  average of the class probabilities predictions of `trees`."
  [trees v]
  (average-values (map #(p/predict-prob % v) trees)))

(defn- predict-log-prob
  "Predict the log probabilities of `v` belonging to each class using
  average of the class log probabilities predictions of `trees`."
  [trees v]
  (average-values (map #(p/predict-log-prob % v) trees)))

(defn- predict
  "Predict the class of `v` from the highest average class probability
  from `trees`."
  [trees v]
  (first (apply max-key second (predict-prob trees v))))

(defrecord RandomForest [trees]
  clojure.lang.IFn
  (invoke [this v] (predict trees v))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args))

  p/ClassProbabilityEstimator
  (predict-prob [this v] (predict-prob trees v))
  (predict-log-prob [this v] (predict-log-prob trees v)))

(defn random-forest
  "Train a random forest of decision trees.

  The training set for each is generated via bootstrapping (sampling with
  replacement).  For each split, a subset of `(sqrt n-features)` features
  are chosen for consideration by sampling without replacement. Predictions
  are made by averaging the sample's predicted class probabilities across
  all of the trees and choosing the class with the highest probability.

  `X` is a samples-x-features feature matrix.  `y` is a vector of class
  labels. `n-trees` (optional, default 10) is the number of trees to use
  in the ensemble.  `max-depth` (optional, default 5) gives the maximum
  height of the trees."
  [X y & {:keys [n-trees max-depth]
          :or {n-trees 10 max-depth 5}}]
  (let [trees (train-forest X y n-trees max-depth)]
    (RandomForest. trees)))

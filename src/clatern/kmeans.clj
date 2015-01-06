(ns clatern.kmeans
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clojure.core.matrix.operators :as M]))

;; kMeans Classification
;; =====================
;;  X : input data
;;  k : number of clusters
;;  num-iters : number of iterations

(defn- cost-fn [X mu m]
  (M/* (/ 1 m) (pow (M/- X mu) 2)))

(defn- rand-centroids
  "Assign initial random centroids"
  [X k lim]
  (map #(get-row X %) (take k (repeatedly #(rand-int lim)))))

(defn- calculate-distances [X centroid]
  (map #(distance % centroid) X))

(defn- calc-centroid [vals]
  (let [n (count vals)]
    (M// (map #(reduce + %) (transpose vals)) n)))

(defn- move-centroids
  "Move centroids to new position"
  [X output centroids]
  (let [indexed-op (map-indexed vector output)]
    (for [i (map first centroids)]
      (let [indices (map first (filter #(= (last %) i) indexed-op))
            X-this (map #(nth X %) indices)]
        (conj [] i (calc-centroid X-this))))))

(defn- cluster-assign
  "Assign clusters to all data items"
  [X centroids]
  (let [cs (map last (seq centroids))
        distances (map #(calculate-distances X %) cs)]
    (map #(first (apply min-key second (map-indexed vector %))) (columns distances))))

(defn- assign-cluster
  "Assign cluster for a new vector"
  [centroids v]
  (let [cs (map last (seq centroids))
        distances (calculate-distances cs v)]
    (first (apply min-key second (map-indexed vector distances)))))

(defrecord kMeans [centroids]
  clojure.lang.IFn
  (invoke [this v] (assign-cluster centroids v))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn kmeans [X & {:keys [k num-iters]
                         :or {k 3
                              num-iters 100}}]
  (let [n (column-count X)
        lim (row-count X)
        init-centroids (map-indexed vector (rand-centroids X k lim))
        centroids (loop [i 0 centroids init-centroids output (repeat lim 0)]
                   (if (= i num-iters)
                     centroids
                     (let [new-output (cluster-assign X centroids)
                           new-centroids (move-centroids X new-output centroids)]
                       (recur (inc i) new-centroids new-output))))]
    (kMeans. centroids)))

(ns clatern.kmeans
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clojure.core.matrix.operators :as M]
            [clatern.implementations :as imp]
            [clatern.protocols :as cp]))

(defn- cost-fn [X mu m]
  (M/* (/ 1 m) (pow (M/- X mu) 2)))

(defn- rand-centroids [X k lim]
  (map #(get-row X %) (take k (repeatedly #(rand-int lim)))))

(defn- calculate-distances [X centroid]
  (pow (map #(reduce + %)
            (pow (sub X centroid)
                 2))
       0.5))

(defn- cluster-assign [X centroids]
  (let [cs (map last (seq centroids))
        distances (map #(calculate-distances X %) cs)]
    (map #(first (apply min-key second (map-indexed vector %))) (columns distances))))

(defn- calc-centroid [vals]
  (let [n (count vals)]
    (M// (map #(reduce + %) (transpose vals)) n)))

(defn- move-centroids [X output centroids]
  (let [indexed-op (map-indexed vector output)]
    (for [i (map first centroids)]
      (let [indices (map first (filter #(= (last %) i) indexed-op))
            X-this (map #(nth X %) indices)]
        (conj [] i (calc-centroid X-this))))))

(defrecord kMeans [data params]
  cp/Model
  (implementation-key [m] :kmeans)

  (set-options [m options]
    (assoc m :params (merge (:params m) options)))
  
  (fit [m new-data]
    (let [X (transpose (:columns new-data))
          cnames (conj (:column-names new-data) "output")
          n (column-count X)
          lim (row-count X)
          k (:n-clusters (:params m))
          centroids (map-indexed vector (rand-centroids X k lim))
          num-iters (:num-iters (:params m))
          centroids (loop [i 0 centroids centroids output (repeat lim 0)]
                   (if (= i num-iters)
                     centroids
                     (let [new-output (cluster-assign X centroids)
                           new-centroids (move-centroids X new-output centroids)]
                       (recur (inc i) new-centroids new-output))))]
      (assoc m :params (assoc (:params m) :centroids centroids))))

  (predict [m new-data]
    (let [X (transpose (:columns new-data))
          cnames (conj (:column-names new-data) "output")
          centroids (:centroids (:params m))
          output (cluster-assign X centroids)]
      (dataset cnames (map conj X output)))))

(imp/register-implementation (kMeans. nil {:num-iters 100,
                                           :n-clusters 5}))

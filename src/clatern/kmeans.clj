(ns clatern.kmeans
  (:require [clojure.set :refer [difference]]
            [clojure.core.matrix :refer :all]
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

(defn- assign-cluster
  "Assign cluster for a new vector"
  [centroids v]
  (let [centroid-distances (for [[idx centroid] centroids] [idx (distance centroid v)])]
  (apply min-key second centroid-distances)))

(defn- cluster-assign
  "Assign clusters to all data items"
  [X centroids]
  (map #(assign-cluster centroids %) (rows X)))

(defn- reassign-if-necessary
  "Reassigns outlier points to empty clusters so that no clusters are empty"
  [X assignments k]
  ; assignments is seq of [cluster-idx distance]
  (let [assigned-cluster-ids (set (map first assignments))]
    (if (= (count assigned-cluster-ids) k)
      assignments
            ; seq of [pt-idx [cluster-idx distance]]
      (let [indexed-assignments (map-indexed vector assignments)
            ; sorted by dist descending
            sorted-assignments (sort-by #(- (last (last %))) indexed-assignments)
            missing-cluster-ids (difference (set (range k)) assigned-cluster-ids)
            n-missing (count missing-cluster-ids)
            point-ids-to-reassign (map first (take n-missing sorted-assignments))
            reassigned (into {} (map vector point-ids-to-reassign missing-cluster-ids))
            ; replace elements in original assignments
            indexed-reassignments (map #(if (contains? reassigned (first %))
                                            [(first %) [(reassigned (first %)) 0.0]] %)
                                         indexed-assignments)
            reassignments (map last indexed-reassignments)]
        reassignments))))
            

(defrecord kMeans [centroids]
  clojure.lang.IFn
  (invoke [this v] (first (assign-cluster centroids v)))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn kmeans [X & {:keys [k num-iters]
                         :or {k 3
                              num-iters 100}}]
  (let [n (column-count X)
        lim (row-count X)
        init-centroids (map-indexed vector (rand-centroids X k lim))
        centroids (loop [i 0 centroids init-centroids]
                    (if (= i num-iters)
                      centroids
                      (let [assignments (cluster-assign X centroids)
                            reassignments (map first (reassign-if-necessary X assignments k))
                            new-centroids (move-centroids X reassignments centroids)]
                        (recur (inc i) new-centroids))))]
    (kMeans. centroids)))

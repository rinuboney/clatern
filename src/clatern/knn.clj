(ns clatern.knn
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clatern.metrics :refer [euclidean-distance]]))

;; k Nearest Neighbour
;; ===================
;;  X : input data
;;  y : target data
;;  v : new input to be classified
;;  k : number of neighbours

(defn- predict [X y v k distance]
  (->> (map vector (map  #(distance % v) X)  y)       
       (sort-by first)
       (take k)
       (map last)
       (frequencies)
       (sort-by val)
       (last)
       (first)))

(defrecord kNN [X y k distance-metric]
  clojure.lang.IFn
  (invoke [this v] (predict X y v k distance-metric))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn knn [X y & {:keys [k distance-metric]
                      :or {k 3
                           distance-metric euclidean-distance}}]
  (kNN. X y k distance-metric))
  

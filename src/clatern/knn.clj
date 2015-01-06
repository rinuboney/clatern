(ns clatern.knn
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]))

;; k Nearest Neighbour
;; ===================
;;  X : input data
;;  y : target data
;;  v : new input to be classified
;;  k : number of neighbours

(defn knn [X y v & {:keys [k]
                      :or {k 3}}]
  (->> (map vector (map  #(distance % v) X)  y)       
       (sort-by first)
       (take k)
       (map last)
       (frequencies)
       (sort-by val)
       (last)
       (first)))

(ns clatern.knn
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]))

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

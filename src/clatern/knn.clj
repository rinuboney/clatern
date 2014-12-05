(ns clatern.knn
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]))

(defn- calculate-distances [X new]
  (pow (map #(reduce + %)
            (pow (sub X new)
                 2))
       0.5))

(defn knn [X y new & {:keys [k]
                      :or {k 3}}]
  (->> (map vector (calculate-distances X new) y)       
       (sort-by first)
       (take k)
       (map last)
       (frequencies)
       (sort-by val)
       (last)
       (first)))

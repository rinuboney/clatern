(ns clatern.knn
  (:use [clojure.core.matrix]
        [clojure.core.matrix.dataset])
  (:require [clatern.protocols :as cp]
            [clatern.implementations :as imp]))

(defn- calculate-distances [X new]
  (pow (map #(reduce + %)
            (pow (sub (transpose X) new)
                 2))
       0.5))

(defn- classify [X y new k]
  (->> (map vector (calculate-distances X new) y)       
       (sort-by first)
       (take k)
       (map last)
       (frequencies)
       (sort-by val)
       (last)
       (first)))

(defrecord kNN [params data]
  cp/Model
  (implementation-key [m] :knn)
  
  (fit [m new-data]
    (if (nil? (:data m))
      (assoc m :data new-data)
      (assoc m :data (join-rows (:data m) new-data))))
  
  (predict [m new-data]
    (let [X (:columns (except-columns (:data m) ["output"]))
          y (get-row (:columns (select-columns (:data m) ["output"])) 0)
          k (:k (:params m))
          new-dmat (select-columns new-data (:column-names (except-columns (:data m) ["output"])))
          cnames (conj (:column-names new-dmat) "output")]
      (dataset cnames
               (map #(conj % (classify X y % k))
                    (rows (transpose (:columns new-dmat))))))))

(imp/register-implementation (kNN. {:k 3} nil))

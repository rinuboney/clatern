(ns clatern.preprocessing
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clojure.core.matrix.operators :as M]
            [clojure.core.matrix.stats :refer :all]))

(defn mean-normalizer
  "Create a mean normalization encoder from the dataset/matrix ds for the specified indices"
  [ds indices]
  (if (= :all indices)
    (let [mu (mean ds)
          std (sd ds)]
      (fn [v] (div (sub v mu) std)))
    (let [ds' (transpose (mapv #(get-column ds %) indices))
          mu (mean ds')
          std (sd ds')]
      (fn [v]
        (let [v' (map v indices)
              normalized (div (sub v' mu) std)]
          (set-indices v indices normalized))))))

(defn min-max-scaler
  "Create a min max scaling encoder from the dataset/matrix ds for the specified indices"
  [ds indices]
  (if (= :all indices)
    (let [mins (map emin (columns ds))
          maxs (map emax (columns ds))]
      (fn [v] (div (sub v mins) (sub maxs mins))))
    (let [ds' (transpose (mapv #(get-column ds %) indices))
          mins (map emin (columns ds'))
          maxs (map emax (columns ds'))]
      (fn [v]
        (let [v' (map v indices)
              scaled (div (sub v' mins) (sub maxs mins))]
          (set-indices v indices scaled))))))

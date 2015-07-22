(ns clatern.metrics
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clojure.core.matrix.operators :as M]))

;; Distance / Similarity metrics and measures

(defn euclidean-distance
  "Euclidean distance between x and y"
  [x y]
  (distance x y))

(defn squared-euclidean-distance
  "Squared Euclidean distance between x and y"
  [x y]
  (let [diff (sub x y)]
    (dot diff diff)))

(defn cosine-distance
  "Cosine distance between x and y. Defined as 1.0 - cosine similarity."
  [x y]
  (let [x_unit (normalise x)
        y_unit (normalise y)
        cos_simil (dot x_unit y_unit)]
    (- 1.0 cos_simil)))

(defn manhattan-distance
  "Manhattan (L1, city-block) distance between x and y"
  [x y]
  (let [diff (sub x y)]
    (esum (abs diff))))
